import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from verl.protocol import DataProto
from verl.trainer.opsd_config import OPSDConfig
from verl.trainer.opsd_trainer import RaySimpleGRPOOPSDTrainer
from verl.trainer.ray_trainer import RayPPOTrainer
from verl.trainer.rubric_evolution import EvolutionTarget, EvolvedCriterion, RubricEvolver
from verl.workers.actor.dp_opsd_actor import DataParallelOPSDActor


class FakeBatch:
    def __init__(self, non_tensor_batch):
        self.non_tensor_batch = non_tensor_batch

    def __len__(self):
        return len(self.non_tensor_batch["rubric"])


class RubricGuidedRLSDTest(unittest.TestCase):
    def test_immediate_evolution_updates_current_reward_metrics_and_tensor(self):
        trainer = object.__new__(RaySimpleGRPOOPSDTrainer)
        trainer.rubric_evolver = RubricEvolver(OPSDConfig())
        trainer._reduce_reward_metrics_for_logging = lambda values, **_: {
            "mock/overall": float(np.mean(values["overall"]))
        }

        parent = {
            "title": "Parent",
            "description": "Parent criterion.",
            "weight": 4.0,
            "criterion_id": "parent_0",
            "parent_id": "",
            "source": "parent",
        }
        child = {
            "title": "Child",
            "description": "New current-step criterion.",
            "weight": 4.0,
            "criterion_id": "child_1",
            "parent_id": "parent_0",
            "source": "decompose",
        }
        active_parent = {**parent, "weight": 0.0}
        base_rubric_batch = np.array([[parent], [parent]], dtype=object)
        active_rubric_batch = np.array(
            [[active_parent, child], [active_parent, child]],
            dtype=object,
        )
        batch = DataProto.from_dict(
            tensors={
                "responses": torch.tensor([[11, 12], [21, 0]]),
                "response_mask": torch.tensor([[1, 1], [1, 0]]),
                "token_level_scores": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            },
            non_tensors={
                "rubric": active_rubric_batch,
                "rubric_list": np.array([None, None], dtype=object),
                "prompt_text": np.array(["Q1", "Q2"], dtype=object),
            },
        )
        evolved_metrics = {
            "overall": [0.0, 1.0],
            "accuracy": [0.0, 1.0],
            "criterion_0_score": [0.0, 1.0],
            "criterion_count": [1.0, 1.0],
        }
        remote_result = (torch.zeros((2, 2)), evolved_metrics)
        trainer.reward_fn = SimpleNamespace(
            compute_reward=SimpleNamespace(remote=mock.Mock(return_value=remote_result))
        )
        reward_metric_values = {
            "overall": [1.0, 0.0],
            "accuracy": [1.0, 0.0],
            "criterion_0_score": [1.0, 0.0],
            "criterion_count": [1.0, 1.0],
        }
        metrics = {}

        with mock.patch("verl.trainer.opsd_trainer.ray.get", side_effect=lambda value: value):
            trainer._merge_immediate_evolved_rewards(
                batch,
                base_rubric_batch=base_rubric_batch,
                reward_metric_values=reward_metric_values,
                metrics=metrics,
                timing_raw={},
            )

        self.assertEqual(reward_metric_values["criterion_1_score"], [0.0, 1.0])
        self.assertEqual(reward_metric_values["criterion_count"], [2.0, 2.0])
        self.assertAlmostEqual(reward_metric_values["overall"][0], 0.0)
        self.assertAlmostEqual(reward_metric_values["overall"][1], 1.0)
        self.assertAlmostEqual(batch.batch["token_level_scores"][0, 1].item(), 0.0)
        self.assertAlmostEqual(batch.batch["token_level_scores"][1, 0].item(), 1.0)
        self.assertEqual(metrics["rubric_evolution/immediate_reward_samples"], 2.0)
        teacher_hints = trainer._build_rubric_guided_hints(batch)
        self.assertIn("New current-step criterion.", teacher_hints[0][0])

    def test_batched_proposal_marks_parent_by_identity_after_child_insertion(self):
        trainer = object.__new__(RaySimpleGRPOOPSDTrainer)
        trainer.opsd_config = OPSDConfig(rubric_evolution_max_subcriteria=2)
        trainer.rubric_evolver = RubricEvolver(trainer.opsd_config)
        parent = [
            {"title": "A", "description": "Parent A.", "weight": 5.0},
            {"title": "B", "description": "Parent B.", "weight": 3.0},
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "prompt_text": np.array(["Question"], dtype=object),
                "rubric": np.array([parent], dtype=object),
                "rubric_list": np.array([None], dtype=object),
            },
        )
        trainer.rubric_evolver.apply_active_rubric_to_batch(batch)
        prompt_key = str(batch.non_tensor_batch["rubric_evolution_prompt_key"][0])
        state = trainer.rubric_evolver.store.get(prompt_key)
        assert state is not None
        state.candidates.append(
            EvolvedCriterion(
                criterion_id="child_a",
                parent_index=0,
                parent_criterion_id="parent_0",
                title="A child",
                description="Child inserted before parent B.",
                weight=1.0,
                source="decompose",
                status="active",
                proposed_step=1,
                verify_history=[{"passed": True}],
            )
        )
        target = EvolutionTarget(
            prompt_key=prompt_key,
            prompt_text="Question",
            parent_index=1,
            parent_item=state.parent_rubric[1],
            state="dead",
            pass_rate=0.0,
            group_indices=[0],
            group_responses=["Answer"],
        )

        hint = trainer._build_rubric_evolution_batch_proposal_hint([target])

        self.assertIn("criterion_index=2 [SELECTED AS p1]; title=B", hint)
        self.assertIn("criterion_index=1; title=A child", hint)
        self.assertNotIn("criterion_index=1 [SELECTED AS p1]", hint)

    def test_batched_proposal_hint_lists_exact_required_parent_ids_and_count(self):
        trainer = object.__new__(RaySimpleGRPOOPSDTrainer)
        trainer.opsd_config = OPSDConfig(
            rubric_evolution_max_subcriteria=2,
            rubric_evolution_max_refinements=2,
        )
        trainer.rubric_evolver = SimpleNamespace(
            store=SimpleNamespace(get=lambda _: None)
        )
        parent = {
            "title": "Parent",
            "description": "Parent criterion.",
            "weight": 5.0,
        }
        targets = [
            EvolutionTarget(
                prompt_key="question",
                prompt_text="Question",
                parent_index=2,
                parent_item=parent,
                state="saturated",
                pass_rate=1.0,
                group_indices=[0],
                group_responses=["Answer"],
            ),
            EvolutionTarget(
                prompt_key="question",
                prompt_text="Question",
                parent_index=5,
                parent_item=parent,
                state="dead",
                pass_rate=0.0,
                group_indices=[0],
                group_responses=["Answer"],
            ),
        ]

        hint = trainer._build_rubric_evolution_batch_proposal_hint(targets)

        self.assertIn("Expected parent count: 2", hint)
        self.assertIn(
            'Expected Parent IDs in exact output order: ["p2", "p5"]',
            hint,
        )
        self.assertIn(
            'Expected operations by Parent ID: {"p2": "refine", "p5": "decompose"}',
            hint,
        )
        self.assertIn("exact Parent ID sequence once each", hint)
        self.assertIn("do not return a candidate_id field", hint)

    def test_batched_proposal_template_has_no_concrete_example_id(self):
        template_path = (
            Path(__file__).parents[2]
            / "prompts"
            / "rubric_evolution_batch_proposal.jinja"
        )
        template = template_path.read_text(encoding="utf-8")

        self.assertNotIn('"parent_id": "p2"', template)
        self.assertNotIn('"candidate_id": "p2_c0"', template)
        self.assertNotIn('"candidate_id"', template)
        self.assertIn("no example Parent ID is provided", template)
        self.assertIn("Do not return a candidate_id field", template)

    def test_batched_meta_verifier_hint_lists_exact_required_ids_and_count(self):
        trainer = object.__new__(RaySimpleGRPOOPSDTrainer)
        trainer.rubric_evolver = SimpleNamespace(
            store=SimpleNamespace(get=lambda _: None)
        )
        parent = {
            "title": "Parent",
            "description": "Parent criterion.",
            "weight": 5.0,
        }
        target_2 = EvolutionTarget(
            prompt_key="question",
            prompt_text="Question",
            parent_index=2,
            parent_item=parent,
            state="saturated",
            pass_rate=1.0,
            group_indices=[0],
            group_responses=["Answer"],
        )
        target_5 = EvolutionTarget(
            prompt_key="question",
            prompt_text="Question",
            parent_index=5,
            parent_item=parent,
            state="dead",
            pass_rate=0.0,
            group_indices=[0],
            group_responses=["Answer"],
        )

        hint = trainer._build_rubric_evolution_batch_verifier_hint(
            [
                {
                    "candidate_id": "p2_c0",
                    "target": target_2,
                    "candidate": {"title": "C1", "description": "Candidate one."},
                },
                {
                    "candidate_id": "p5_c1",
                    "target": target_5,
                    "candidate": {"title": "C2", "description": "Candidate two."},
                },
            ]
        )

        self.assertIn("Expected candidate count: 2", hint)
        self.assertIn(
            'Expected Candidate IDs in exact output order: ["p2_c0", "p5_c1"]',
            hint,
        )
        self.assertIn("exact ID sequence once each", hint)

    def test_batched_meta_verifier_template_has_no_concrete_example_id(self):
        template_path = (
            Path(__file__).parents[2]
            / "prompts/rubric_evolution_batch_verify.jinja"
        )
        template = template_path.read_text(encoding="utf-8")

        self.assertNotIn('"candidate_id": "p2_c0"', template)
        self.assertIn("no example Candidate ID is provided", template)
        self.assertIn(
            '"target_relevance": {"verdict": "PASS|FAIL|UNCERTAIN", "reason"',
            template,
        )

    def test_trainer_checkpoint_persists_and_restores_evolved_reward_rubric(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir) / "global_step_7"
            checkpoint_dir.mkdir()
            trainer_config = SimpleNamespace(
                trainer=SimpleNamespace(
                    save_checkpoint_path=tmp_dir,
                    load_checkpoint_path=None,
                    nnodes=1,
                    n_gpus_per_node=1,
                ),
                worker=SimpleNamespace(actor=SimpleNamespace(separate_teacher_model=True)),
            )

            trainer = object.__new__(RaySimpleGRPOOPSDTrainer)
            trainer.config = trainer_config
            trainer.global_step = 7
            trainer.rubric_evolver = RubricEvolver(OPSDConfig())
            parent = [{"title": "Parent", "description": "Parent criterion.", "weight": 5.0}]
            batch = DataProto.from_dict(
                tensors={},
                non_tensors={
                    "prompt_text": np.array(["Question"], dtype=object),
                    "rubric": np.array([parent], dtype=object),
                    "rubric_list": np.array([None], dtype=object),
                },
            )
            trainer.rubric_evolver.apply_active_rubric_to_batch(batch)
            prompt_key = str(batch.non_tensor_batch["rubric_evolution_prompt_key"][0])
            state = trainer.rubric_evolver.store.get(prompt_key)
            assert state is not None
            state.candidates.append(
                EvolvedCriterion(
                    criterion_id="child_saved",
                    parent_index=0,
                    parent_criterion_id="parent_0",
                    title="Saved child",
                    description="Participates in reward after resume.",
                    weight=1.0,
                    source="decompose",
                    status="active",
                    proposed_step=4,
                    verify_history=[{"passed": True}],
                )
            )

            from liverubric.checkpoints import required_files
            for name in required_files(1, teacher=True, initialization=False, evolution=False):
                path = checkpoint_dir / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test worker state")
            trainer._save_extra_checkpoint_state(str(checkpoint_dir))

            state_path = checkpoint_dir / trainer.RUBRIC_EVOLUTION_CHECKPOINT
            self.assertTrue(state_path.exists())
            self.assertEqual(json.loads(state_path.read_text())["version"], 1)

            restored = object.__new__(RaySimpleGRPOOPSDTrainer)
            restored.config = SimpleNamespace(
                trainer=SimpleNamespace(
                    save_checkpoint_path=tmp_dir,
                    load_checkpoint_path=str(checkpoint_dir),
                    nnodes=1,
                    n_gpus_per_node=1,
                ),
                worker=SimpleNamespace(actor=SimpleNamespace(separate_teacher_model=True)),
            )
            restored.global_step = 7
            restored.rubric_evolver = RubricEvolver(OPSDConfig())
            restored._load_extra_checkpoint_state(str(checkpoint_dir))

            restored_state = restored.rubric_evolver.store.get(prompt_key)
            assert restored_state is not None
            self.assertEqual(
                [item["criterion_id"] for item in restored_state.build_active_rubric()],
                ["parent_0", "child_saved"],
            )

    def test_evolution_logging_serializes_structured_cells_and_is_nonfatal(self):
        class FakeTable:
            def __init__(self, columns):
                self.columns = columns
                self.rows = []

            def add_data(self, *values):
                self.rows.append(values)

        events = [
            {
                "step": 1,
                "outcome": "rejected",
                "candidate_score_vector": [1.0, 0.0],
                "semantic_failed_dimensions": ["target_relevance"],
                "semantic_uncertain_dimensions": [],
                "semantic_verify_checks": {
                    "target_relevance": {
                        "verdict": "FAIL",
                        "reason": "The candidate adds an unsupported number.",
                    }
                },
                "collision_ids": ["parent_0"],
            },
            {
                "step": 1,
                "outcome": "rejected",
                "candidate_score_vector": None,
                "semantic_failed_dimensions": [],
                "semantic_uncertain_dimensions": [],
                "semantic_verify_checks": {},
                "collision_ids": None,
            },
        ]
        fake_wandb = SimpleNamespace(
            Table=FakeTable,
            Html=lambda value: value,
            log=mock.Mock(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = object.__new__(RaySimpleGRPOOPSDTrainer)
            trainer.config = SimpleNamespace(
                trainer=SimpleNamespace(
                    logger=["wandb"],
                    save_checkpoint_path=tmp_dir,
                )
            )
            trainer.rubric_evolver = SimpleNamespace(
                pop_recent_evolution_examples=mock.Mock(return_value=events)
            )

            with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
                trainer._log_rubric_evolution_wandb_table(step=1)

                payload = fake_wandb.log.call_args.args[0]
                table = payload["rubric_evolution/examples"]
                structured_columns = [
                    "candidate_score_vector",
                    "semantic_failed_dimensions",
                    "semantic_uncertain_dimensions",
                    "semantic_verify_checks",
                    "collision_ids",
                ]
                for column in structured_columns:
                    column_index = table.columns.index(column)
                    self.assertTrue(
                        all(isinstance(row[column_index], str) for row in table.rows)
                    )
                checks_index = table.columns.index("semantic_verify_checks")
                self.assertEqual(
                    json.loads(table.rows[0][checks_index]),
                    events[0]["semantic_verify_checks"],
                )
                self.assertEqual(json.loads(table.rows[1][checks_index]), {})

                fake_wandb.log.side_effect = RuntimeError("wandb unavailable")
                with mock.patch("builtins.print") as warning:
                    trainer._log_rubric_evolution_wandb_table(step=2)
                self.assertIn("training continues", warning.call_args.args[0])

            jsonl_path = Path(tmp_dir) / "rubric_evolution_cases.jsonl"
            saved_events = [
                json.loads(line) for line in jsonl_path.read_text().splitlines()
            ]
            self.assertEqual(saved_events, events + events)

    def test_config_requires_simple_rlsd(self):
        config = OPSDConfig(enable_rubric_guided_rlsd=True)

        with self.assertRaisesRegex(ValueError, "enable_grpo_opsd_simple"):
            config.post_init()

    def test_config_is_mutually_exclusive_with_casd(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_casd=True,
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            config.post_init()

    def test_config_requires_dedicated_teacher_template(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
        )

        with self.assertRaisesRegex(ValueError, "rubric_guided_teacher_template"):
            config.post_init()

    def test_dead_criteria_guided_rollout_requires_group_size(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_dead_criteria_guided_rollout=True,
            dead_criteria_guided_rollout_n=1,
        )

        with self.assertRaisesRegex(ValueError, "dead_criteria_guided_rollout_n"):
            config.post_init()

    def test_complete_rubric_builds_one_privileged_hint_per_sample(self):
        trainer = object.__new__(RaySimpleGRPOOPSDTrainer)
        batch = FakeBatch(
            {
                "rubric": np.array(
                    [
                        [
                            {
                                "title": "Completeness",
                                "description": "Mentions medical supervision.",
                                "weight": 9,
                            },
                            {
                                "title": "Safety",
                                "description": "Fails to mention urgent warning signs.",
                                "weight": -8,
                            },
                        ]
                    ],
                    dtype=object,
                ),
                "rubric_list": None,
            }
        )

        hints = trainer._build_rubric_guided_hints(batch)

        self.assertEqual(len(hints), 1)
        self.assertEqual(len(hints[0]), 1)
        self.assertIn("Requirements to satisfy:", hints[0][0])
        self.assertIn("Penalty conditions to avoid:", hints[0][0])
        self.assertIn("Mentions medical supervision.", hints[0][0])
        self.assertIn("Fails to mention urgent warning signs.", hints[0][0])

    def test_dead_criteria_guided_targets_include_all_dead_positive_criteria(self):
        trainer = object.__new__(RaySimpleGRPOOPSDTrainer)
        trainer.opsd_config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_dead_criteria_guided_rollout=True,
        )
        batch = FakeBatch(
            {
                "uid": np.array(["a", "a", "b", "b"], dtype=object),
                "rubric": np.array(
                    [
                        [
                            {"title": "A", "description": "Mentions A.", "weight": 2},
                            {"title": "B", "description": "Mentions B.", "weight": 3},
                            {"title": "Penalty", "description": "Avoids C.", "weight": -1},
                        ],
                        [
                            {"title": "A", "description": "Mentions A.", "weight": 2},
                            {"title": "B", "description": "Mentions B.", "weight": 3},
                            {"title": "Penalty", "description": "Avoids C.", "weight": -1},
                        ],
                        [
                            {"title": "A", "description": "Mentions A.", "weight": 2},
                            {"title": "B", "description": "Mentions B.", "weight": 3},
                            {"title": "Penalty", "description": "Avoids C.", "weight": -1},
                        ],
                        [
                            {"title": "A", "description": "Mentions A.", "weight": 2},
                            {"title": "B", "description": "Mentions B.", "weight": 3},
                            {"title": "Penalty", "description": "Avoids C.", "weight": -1},
                        ],
                    ],
                    dtype=object,
                ),
                "rubric_list": None,
            }
        )
        reward_metric_values = {
            "criterion_0_score": [0.0, 0.0, 0.0, 1.0],
            "criterion_1_score": [0.0, 0.0, 0.0, 0.0],
            "criterion_2_score": [0.0, 0.0, 0.0, 0.0],
        }

        targets = trainer._build_dead_criteria_guided_targets(batch, reward_metric_values)

        self.assertEqual(len(targets), 2)
        self.assertEqual([item["title"] for item in targets[0]["dead_items"]], ["A", "B"])
        self.assertEqual([item["title"] for item in targets[1]["dead_items"]], ["B"])
        self.assertNotIn("Penalty", targets[0]["hint"])

    def test_dead_criteria_guided_targets_include_triggered_negative_penalties(self):
        trainer = object.__new__(RaySimpleGRPOOPSDTrainer)
        trainer.opsd_config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_dead_criteria_guided_rollout=True,
        )
        batch = FakeBatch(
            {
                "uid": np.array(["a", "a", "b", "b"], dtype=object),
                "rubric": np.array(
                    [
                        [
                            {"title": "A", "description": "Mentions A.", "weight": 2},
                            {"title": "Penalty", "description": "Gives unsafe advice.", "weight": -4},
                        ],
                        [
                            {"title": "A", "description": "Mentions A.", "weight": 2},
                            {"title": "Penalty", "description": "Gives unsafe advice.", "weight": -4},
                        ],
                        [
                            {"title": "A", "description": "Mentions A.", "weight": 2},
                            {"title": "Penalty", "description": "Gives unsafe advice.", "weight": -4},
                        ],
                        [
                            {"title": "A", "description": "Mentions A.", "weight": 2},
                            {"title": "Penalty", "description": "Gives unsafe advice.", "weight": -4},
                        ],
                    ],
                    dtype=object,
                ),
                "rubric_list": None,
            }
        )
        reward_metric_values = {
            "criterion_0_score": [0.0, 0.0, 0.0, 0.0],
            "criterion_1_score": [1.0, 1.0, 1.0, 0.0],
        }

        targets = trainer._build_dead_criteria_guided_targets(batch, reward_metric_values)

        self.assertEqual(len(targets), 2)
        self.assertEqual([item["title"] for item in targets[0]["dead_items"]], ["A", "Penalty"])
        self.assertEqual([item["title"] for item in targets[1]["dead_items"]], ["A"])
        self.assertIn("Penalty conditions to avoid:", targets[0]["hint"])
        self.assertIn("Gives unsafe advice.", targets[0]["hint"])

    def test_dead_criteria_guided_max_group_ratio_limits_small_batches_to_zero(self):
        trainer = object.__new__(RaySimpleGRPOOPSDTrainer)
        trainer.opsd_config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_dead_criteria_guided_rollout=True,
            dead_criteria_guided_max_group_ratio=0.25,
        )
        batch = FakeBatch(
            {
                "uid": np.array(["a", "b", "c"], dtype=object),
                "rubric": np.array(
                    [
                        [{"title": "A", "description": "Mentions A.", "weight": 2}],
                        [{"title": "A", "description": "Mentions A.", "weight": 2}],
                        [{"title": "A", "description": "Mentions A.", "weight": 2}],
                    ],
                    dtype=object,
                ),
                "rubric_list": None,
            }
        )
        reward_metric_values = {"criterion_0_score": [0.0, 0.0, 0.0]}

        targets = trainer._build_dead_criteria_guided_targets(batch, reward_metric_values)

        self.assertEqual(targets, [])

    def test_log_criterion_scores_filters_logger_metrics_only(self):
        trainer = object.__new__(RaySimpleGRPOOPSDTrainer)
        trainer.config = SimpleNamespace(
            worker=SimpleNamespace(
                reward=SimpleNamespace(
                    log_criterion_scores=False,
                    reward_function_kwargs={},
                )
            )
        )
        reward_metrics = {
            "overall": [0.0, 1.0],
            "accuracy": [0.0, 1.0],
            "criterion_0_score": [0.0, 1.0],
            "criterion_count": [2.0, 2.0],
        }

        logged = trainer._reduce_reward_metrics_for_logging(reward_metrics, prefix="reward/")

        self.assertIn("reward/overall", logged)
        self.assertIn("reward/accuracy", logged)
        self.assertNotIn("reward/criterion_0_score", logged)
        self.assertNotIn("reward/criterion_count", logged)
        self.assertIn("criterion_0_score", reward_metrics)

    def test_split_with_remainder_keeps_tail_batch(self):
        batch = DataProto.from_dict(tensors={"x": torch.arange(5)})

        chunks = DataParallelOPSDActor._split_with_remainder(batch, split_size=2)

        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])

    def test_split_with_remainder_can_merge_tail_batch(self):
        batch = DataProto.from_dict(tensors={"x": torch.arange(9)})

        chunks = DataParallelOPSDActor._split_with_remainder(
            batch,
            split_size=4,
            merge_trailing_remainder=True,
        )

        self.assertEqual([len(chunk) for chunk in chunks], [4, 5])
        self.assertTrue(torch.equal(chunks[0].batch["x"], torch.arange(0, 4)))
        self.assertTrue(torch.equal(chunks[1].batch["x"], torch.arange(4, 9)))

    def test_split_with_remainder_does_not_merge_exact_batches(self):
        batch = DataProto.from_dict(tensors={"x": torch.arange(8)})

        chunks = DataParallelOPSDActor._split_with_remainder(
            batch,
            split_size=4,
            merge_trailing_remainder=True,
        )

        self.assertEqual([len(chunk) for chunk in chunks], [4, 4])

    def test_dynamic_batching_accounts_for_longer_teacher_prompts(self):
        student_mask = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1],
            ]
        )
        teacher_mask = torch.ones_like(student_mask)
        batch = DataProto.from_dict(
            tensors={
                "attention_mask": student_mask.clone(),
                "grpo_opsd_teacher_attention_mask": teacher_mask,
            }
        )

        micro_batches = DataParallelOPSDActor._prepare_grpo_opsd_dynamic_batch(
            batch,
            max_token_len=8,
        )

        self.assertEqual(len(micro_batches), 2)
        for micro_batch in micro_batches:
            self.assertEqual(len(micro_batch), 1)
            self.assertEqual(int(micro_batch.batch["attention_mask"].sum()), 2)
        self.assertTrue(torch.equal(batch.batch["attention_mask"], student_mask))


if __name__ == "__main__":
    unittest.main()
