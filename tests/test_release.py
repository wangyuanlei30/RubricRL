"""CPU regression checks for real release modules; no model or judge requests."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import yaml

from liverubric.checkpoints import (atomic_json, publish_checkpoint, required_files,
                                    resolve_checkpoint, validate_checkpoint)
from liverubric.launch import (DEFAULT_CONFIG, ROOT, code_identity, make_plan,
                               sha256, validate_settings, validate_source)
from rubric_generation.offline_question_rubrics import question_messages, validate_completion
from rubric_generation.rubric_initializer import InitialRubricPromptBuilder
from verl.trainer.rubric_generator_rewards import (CandidateAssessment, ProposalAssessment,
    group_relative_advantages, parse_rg_proposal, score_rg_proposal)
from verl.trainer.rubric_weight_allocation import equal_family_weight
from verl.workers.rubric_teacher_state import (save_teacher_state, load_teacher_state,
                                               sync_teacher_parameters)


def worker_fixture(directory, world=2):
    for name in required_files(world, teacher=True, initialization=False, evolution=True):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"state": 1}')


def launch_args(directory, **changes):
    values = dict(model=directory / "model", train_file=directory / "questions.jsonl",
        val_file=None, output=directory / "run", config=DEFAULT_CONFIG,
        gpus=4, init_tp=4, init_dir=None, resume=None)
    values.update(changes)
    return argparse.Namespace(**values)


class RewardTests(unittest.TestCase):
    def proposal(self, p=0.5, passed=True):
        candidate = CandidateAssessment("gap", 0, {"description": "Concrete requirement"},
            semantic={"valid": True}, empirical={"passed": passed, "pass_rate": p})
        return ProposalAssessment(("gap",), [candidate], {"gap": 1}, {"gap": 6})

    def score(self, proposal):
        return score_rg_proposal(proposal, output_tokens=30, token_budget=4096)

    def test_eligible_splits_have_equal_reward(self):
        for p in (0.25, 0.375, 0.5, 0.625, 0.75):
            self.assertEqual(self.score(self.proposal(p))["reward"], 2.0)

    def test_semantic_valid_but_ineligible_has_no_disc_reward(self):
        result = self.score(self.proposal(0.125, False))
        self.assertEqual((result["validity"], result["discrimination"]), (1.0, 0.0))

    def test_rejected_candidates_remain_in_denominator(self):
        proposal = self.proposal()
        proposal.raw_counts["gap"] = 2
        proposal.candidates.append(CandidateAssessment("gap", 1, None, rejection="duplicate"))
        self.assertEqual(self.score(proposal)["reward"], 1.0)

    def test_empty_target_is_in_target_average(self):
        proposal = self.proposal()
        proposal.target_ids += ("empty",)
        proposal.raw_counts["empty"], proposal.budgets["empty"] = 0, 2
        self.assertEqual(self.score(proposal)["reward"], 1.0)

    def test_invalid_format_and_unavailable_audit_differ(self):
        proposal = self.proposal()
        proposal.format_valid = False
        self.assertEqual(self.score(proposal)["reward"], -1.0)
        proposal.audit_error = "timeout"
        with self.assertRaises(ValueError):
            self.score(proposal)

    def test_overbudget_and_duplicate_slots_not_dropped(self):
        item = dict(title="Safety", description="Mentions safety", weight=1)
        proposal = parse_rg_proposal(json.dumps(dict(operation="gap_expand", criteria=[item] * 3)),
            role="gap", operations={"gap": "gap_expand"}, budgets={"gap": 2},
            weights={"gap": 1}, existing_rubric=[])
        self.assertEqual(proposal.raw_counts["gap"], 3)
        self.assertEqual(len(proposal.candidates), 3)
        self.assertTrue(proposal.candidates[1].rejection)
        self.assertEqual(proposal.candidates[2].rejection, "over_budget")

    def test_group_advantages_never_cross_prompt_groups(self):
        values = group_relative_advantages([1, 3, 20, 20], ["a", "a", "b", "b"], expected_n=2)
        self.assertAlmostEqual(values[0], -1.0, places=5)
        self.assertEqual(values[2:], [0, 0])
        with self.assertRaises(ValueError):
            group_relative_advantages([1], ["a"], expected_n=2)

    def test_signed_family_weight_is_conserved(self):
        for weight in (-8, 0, 5):
            for count in (0, 1, 2, 4):
                self.assertAlmostEqual(equal_family_weight(weight, count) * (count + 1), weight)


class PromptAndConfigTests(unittest.TestCase):
    def test_original_prompts_are_byte_identical(self):
        manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
        prompts = [item for item in manifest["files"] if item["release"].startswith("prompts/")]
        self.assertGreater(len(prompts), 10)
        for item in prompts:
            if item.get("local_only") and not (ROOT / item["release"]).exists():
                continue
            self.assertEqual(sha256(ROOT / item["release"]), item["source_sha256"], item["release"])

    @unittest.skipUnless((ROOT / "prompts/healthbench_example_level_fewshot.json").exists(),
                         "Prepare the local HealthBench demonstrations to run this test")
    def test_initialization_removes_reference_answers_and_self_demonstration(self):
        builder = InitialRubricPromptBuilder(str(ROOT / "prompts"), few_shot_examples=5)
        examples = json.loads((ROOT / "prompts/healthbench_example_level_fewshot.json").read_text())
        question = examples[0]["messages"][0]["content"].split("### Target Conversation\n", 1)[1].split("\n\n### Reference Answer", 1)[0]
        messages = question_messages(builder, question)
        self.assertEqual(len(messages), 10)  # system + four pairs + target
        self.assertNotIn("### Reference Answer", json.dumps(messages))
        self.assertEqual(sum(question in m["content"] for m in messages if m["role"] == "user"), 1)

    def test_initialization_parser_validates_complete_rubric(self):
        items = [dict(criterion=f"Requirement {i}", points=1 if i else -2) for i in range(7)]
        self.assertEqual(len(validate_completion(json.dumps(items), "Q")), 7)
        with self.assertRaises(ValueError):
            validate_completion(json.dumps(items[:2]), "Q")

    def test_paper_loss_coefficients_and_teacher(self):
        config = yaml.safe_load(DEFAULT_CONFIG.read_text())
        validate_settings(config)
        o = config["opsd"]
        self.assertEqual(o["rubric_generator_loss_coef"] * o["rubric_generator_gap_loss_weight"], .125)
        self.assertEqual(o["rubric_generator_loss_coef"] * o["rubric_generator_vertical_loss_weight"], .125)
        self.assertEqual((o["rlsd_lambda"], o["rlsd_reweight_clip_range"], o["rlsd_lambda_decay_steps"]), (.5, .2, 0))
        self.assertFalse(o["enable_rubric_initialization"])

    def test_config_paths_exist_and_dry_plan_does_not_write(self):
        with tempfile.TemporaryDirectory() as temp:
            args = launch_args(Path(temp))
            plan = make_plan(args)
            for section in (plan["config"]["opsd"], plan["config"]["data"]):
                for key, value in section.items():
                    if key.endswith("template") or key in ("format_prompt", "rubric_initialization_prompt_dir"):
                        self.assertTrue(Path(value).exists(), key)
            self.assertIn("rubric_generation.offline_question_rubrics", plan["init_command"])
            self.assertIn("verl.trainer.opsd_main", plan["train_command"])
            self.assertFalse(args.output.exists())

    def test_source_schema_and_minimum_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.jsonl"
            path.write_text(json.dumps({"prompt": "Q", "answer": ""}) + "\n")
            self.assertEqual(validate_source(path, batch_size=1), sha256(path))
            with self.assertRaises(ValueError):
                validate_source(path, batch_size=64)

    def test_checkpoint_hook_precedes_tracker(self):
        tree = ast.parse((ROOT / "verl/trainer/ray_trainer.py").read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_save_checkpoint")
        calls = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(method) if isinstance(n, ast.Call)]
        line = dict((name, lineno) for lineno, name in calls)
        self.assertLess(line["self._save_extra_checkpoint_state"], line["atomic_json"])
        self.assertLess(line["atomic_json"], line["remove_obsolete_ckpt"])


class CheckpointTests(unittest.TestCase):
    def test_full_state_manifest_roundtrip_and_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "global_step_10"
            worker_fixture(path)
            publish_checkpoint(path, world_size=2, teacher=True, initialization=False, evolution=True)
            validate_checkpoint(path, world_size=2)
            (path / "actor/optim_world_size_2_rank_1.pt").write_text("")
            with self.assertRaises(ValueError):
                validate_checkpoint(path, world_size=2)

    def test_missing_teacher_cannot_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            worker_fixture(path)
            (path / "actor/rubric_teacher/model_world_size_2_rank_1.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                publish_checkpoint(path, world_size=2, teacher=True, initialization=False, evolution=True)
            self.assertFalse((path / "liverubric_checkpoint.json").exists())

    def test_world_size_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            worker_fixture(path)
            publish_checkpoint(path, world_size=2, teacher=True, initialization=False, evolution=True)
            with self.assertRaises(ValueError):
                validate_checkpoint(path, world_size=4)

    def test_latest_follows_tracker_not_incomplete_newer_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "global_step_10"
            worker_fixture(path)
            publish_checkpoint(path, world_size=2, teacher=True, initialization=False, evolution=True)
            atomic_json(root / "checkpoint_tracker.json", {"last_global_step": 10})
            (root / "global_step_20").mkdir()
            self.assertEqual(resolve_checkpoint(root, "latest", world_size=2), path.resolve())

    def test_teacher_save_restore_preserves_its_own_weights(self):
        teacher, actor = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        sync_teacher_parameters(teacher, actor)
        saved = {k: v.clone() for k, v in teacher.state_dict().items()}
        with tempfile.TemporaryDirectory() as temp:
            save_teacher_state(teacher, temp)
            with torch.no_grad():
                teacher.weight.fill_(99)
                actor.weight.fill_(42)
            load_teacher_state(teacher, temp)
            for key, value in teacher.state_dict().items():
                torch.testing.assert_close(value, saved[key])
            self.assertFalse(torch.equal(teacher.weight, actor.weight))

    def test_resume_plan_keeps_saved_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            args = launch_args(Path(temp))
            plan = make_plan(args)
            args.output.mkdir()
            atomic_json(args.output / "resolved_config.json", plan["config"])
            metadata = plan["metadata"]
            metadata["config_sha256"] = sha256(args.output / "resolved_config.json")
            atomic_json(args.output / "launch.json", metadata)
            path = args.output / "checkpoints/global_step_10"
            worker_fixture(path, world=4)
            publish_checkpoint(path, world_size=4, teacher=True, initialization=False, evolution=True)
            atomic_json(path.parent / "checkpoint_tracker.json", {"last_global_step": 10})
            resume = make_plan(launch_args(Path(temp), model=None, train_file=None, resume="latest"))
            self.assertIsNone(resume["init_command"])
            self.assertEqual(resume["config"], plan["config"])
            self.assertIn(f"trainer.load_checkpoint_path={path.resolve()}", resume["train_command"])


if __name__ == "__main__":
    unittest.main()
