import json
import unittest

import numpy as np

from verl.protocol import DataProto
from verl.trainer.opsd_config import OPSDConfig
from verl.trainer.rubric_evolution import (
    CANDIDATE_ACTIVE,
    CRITERION_STATE_DEAD,
    CRITERION_STATE_MIXED,
    CRITERION_STATE_SATURATED,
    EvolvedCriterion,
    RubricEvolver,
    classify_criterion_state,
    classify_pass_rate,
    find_unsupported_numeric_claims,
    make_prompt_key,
    normalize_rubric_items,
    parse_batched_candidate_verifier_payloads,
    parse_batched_proposal_payload,
    parse_candidate_verifier_verdict,
    parse_collision_verifier_verdict,
    parse_collision_pair_verdict,
    parse_negative_polarity_verdict,
    parse_parent_novelty_verdict,
    parse_proposed_criteria,
    validate_proposal_evidence,
)


def _verifier_payload(
    *,
    valid=True,
    parent_anchored=True,
    polarity_preserving=True,
    not_redundant_with_parent=True,
    not_redundant_with_active_rubric=True,
    not_redundant_with_sibling_candidates=True,
    operation_valid=True,
    factual_grounded=True,
    counterexample_found=False,
    behavior_direction_preserved=True,
    required_implication_holds=True,
    judgeable=True,
    reason="ok",
):
    return json.dumps(
        {
            "parent_anchored": parent_anchored,
            "operation_valid": operation_valid,
            "polarity_preserving": polarity_preserving,
            "behavior_direction_preserved": behavior_direction_preserved,
            "required_implication_holds": required_implication_holds,
            "not_paraphrase_of_parent": not_redundant_with_parent,
            "not_redundant_with_active_rubric": not_redundant_with_active_rubric,
            "not_redundant_with_sibling_candidates": not_redundant_with_sibling_candidates,
            "judgeable": judgeable,
            "factual_grounded": factual_grounded,
            "counterexample_found": counterexample_found,
            "valid": valid,
            "reason": reason,
        }
    )


def _multidimensional_result(
    candidate_id="p0_c0",
    *,
    overrides=None,
    final_verdict="VALID",
):
    dimensions = [
        "target_relevance",
        "polarity_consistency",
        "factual_grounding",
        "measurability",
        "novelty_and_non_duplication",
        "specificity_and_non_triviality",
        "style_relevance",
        "rollout_generalization",
        "atomicity",
    ]
    checks = {
        dimension: {"verdict": "PASS", "reason": "check passed"}
        for dimension in dimensions
    }
    for dimension, verdict in (overrides or {}).items():
        checks[dimension] = {"verdict": verdict, "reason": f"{dimension} {verdict.lower()}"}
    return {
        "candidate_id": candidate_id,
        "checks": checks,
        "final_verdict": final_verdict,
        "failed_dimensions": [
            dimension
            for dimension, check in checks.items()
            if check["verdict"] == "FAIL"
        ],
        "final_reason": "model aggregate",
    }


class RubricEvolutionTest(unittest.TestCase):
    def test_config_requires_rubric_guided_rlsd(self):
        config = OPSDConfig(enable_rubric_evolution=True)
        with self.assertRaisesRegex(ValueError, "enable_rubric_guided_rlsd"):
            config.post_init()

    def test_config_rejects_nonpositive_evolution_prompt_budget(self):
        config = OPSDConfig(rubric_evolution_max_prompt_length=0)
        with self.assertRaisesRegex(ValueError, "rubric_evolution_max_prompt_length"):
            config.post_init()

    def test_config_requires_batched_proposal_template_when_enabled(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
        )
        with self.assertRaisesRegex(ValueError, "rubric_evolution_batch_proposal_template"):
            config.post_init()

    def test_config_requires_batched_verifier_template_when_evolution_enabled(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
            rubric_evolution_batch_proposal_template="batch-proposal.jinja",
        )
        with self.assertRaisesRegex(ValueError, "rubric_evolution_batch_verify_template"):
            config.post_init()

    def test_config_does_not_require_legacy_verifier_templates(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_batch_proposal_template="batch-proposal.jinja",
            rubric_evolution_batch_verify_template="batch-verify.jinja",
        )
        config.post_init()

    def test_classify_pass_rate(self):
        self.assertEqual(
            classify_pass_rate(0.0, tau_dead=0.05, tau_sat=0.95),
            CRITERION_STATE_DEAD,
        )
        self.assertEqual(
            classify_pass_rate(0.5, tau_dead=0.05, tau_sat=0.95),
            CRITERION_STATE_MIXED,
        )
        self.assertEqual(
            classify_pass_rate(1.0, tau_dead=0.05, tau_sat=0.95),
            CRITERION_STATE_SATURATED,
        )

    def test_classify_criterion_state_reverses_negative_weight(self):
        negative = {"title": "Pitfall (example)", "description": "Bad behavior.", "weight": -2.0}
        self.assertEqual(
            classify_criterion_state(0.0, negative, tau_dead=0.05, tau_sat=0.95),
            CRITERION_STATE_SATURATED,
        )
        self.assertEqual(
            classify_criterion_state(1.0, negative, tau_dead=0.05, tau_sat=0.95),
            CRITERION_STATE_DEAD,
        )
        self.assertEqual(
            classify_criterion_state(0.5, negative, tau_dead=0.05, tau_sat=0.95),
            CRITERION_STATE_MIXED,
        )

    def test_parse_proposed_criteria(self):
        payload = {
            "subcriteria": [
                {"title": "Exposure", "description": "Asks about TB contacts.", "weight": 2.0}
            ]
        }
        parsed = parse_proposed_criteria(json.dumps(payload), max_items=4)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["description"], "Asks about TB contacts.")

    def test_parse_batched_proposal_routes_only_unique_expected_parents(self):
        payload = {
            "targets": [
                {
                    "parent_id": "p1",
                    "operation": "decompose",
                    "candidates": [
                        {"title": "Exposure", "description": "Asks about TB contacts."}
                    ],
                },
                {"parent_id": "p2", "operation": "refine", "candidates": []},
                {"parent_id": "p99", "operation": "refine", "candidates": []},
            ]
        }
        parsed, diagnostics = parse_batched_proposal_payload(
            json.dumps(payload),
            expected_operations={"p1": "decompose", "p2": "refine"},
            max_items_by_parent={"p1": 2, "p2": 2},
        )
        self.assertTrue(diagnostics["parse_valid"])
        self.assertEqual(parsed["p1"][0]["description"], "Asks about TB contacts.")
        self.assertEqual(parsed["p2"], [])
        self.assertEqual(diagnostics["unknown_parent_ids"], ["p99"])

    def test_parse_batched_proposal_accepts_top_level_targets_array(self):
        payload = [
            {
                "parent_id": "p2",
                "operation": "refine",
                "candidates": [
                    {
                        "title": "More specific",
                        "description": "Adds a stricter response-judgeable distinction.",
                        "operation": "refine",
                    }
                ],
            },
            {"parent_id": "p5", "operation": "decompose", "candidates": []},
        ]

        parsed, diagnostics = parse_batched_proposal_payload(
            json.dumps(payload),
            expected_operations={"p2": "refine", "p5": "decompose"},
            max_items_by_parent={"p2": 2, "p5": 2},
        )

        self.assertTrue(diagnostics["parse_valid"])
        self.assertEqual(diagnostics["missing_parent_ids"], [])
        self.assertEqual(diagnostics["unknown_parent_ids"], [])
        self.assertEqual(len(parsed["p2"]), 1)
        self.assertEqual(parsed["p5"], [])

    def test_parse_batched_proposal_fails_closed_on_duplicate_parent(self):
        payload = {
            "targets": [
                {"parent_id": "p1", "operation": "decompose", "candidates": []},
                {
                    "parent_id": "p1",
                    "operation": "decompose",
                    "candidates": [{"description": "Must not be routed."}],
                },
            ]
        }
        parsed, diagnostics = parse_batched_proposal_payload(
            json.dumps(payload),
            expected_operations={"p1": "decompose"},
            max_items_by_parent={"p1": 2},
        )
        self.assertEqual(parsed["p1"], [])
        self.assertEqual(diagnostics["duplicate_parent_ids"], ["p1"])

    def test_multidimensional_verdict_trusts_model_final_judgment(self):
        model_disagrees = _multidimensional_result(final_verdict="INVALID")
        verdict = parse_candidate_verifier_verdict(json.dumps(model_disagrees))
        self.assertFalse(verdict["valid"])
        self.assertEqual(verdict["model_final_verdict"], "INVALID")

        failed = _multidimensional_result(
            overrides={"factual_grounding": "FAIL"},
            final_verdict="VALID",
        )
        verdict = parse_candidate_verifier_verdict(json.dumps(failed))
        self.assertTrue(verdict["valid"])
        self.assertEqual(verdict["failed_dimensions"], ["factual_grounding"])

    def test_multidimensional_verdict_fails_closed_on_uncertain_or_missing_check(self):
        uncertain = _multidimensional_result(
            overrides={"target_relevance": "UNCERTAIN"},
            final_verdict="REVIEW",
        )
        verdict = parse_candidate_verifier_verdict(json.dumps(uncertain))
        self.assertFalse(verdict["valid"])
        self.assertEqual(verdict["model_final_verdict"], "REVIEW")

        missing = _multidimensional_result()
        del missing["checks"]["atomicity"]
        verdict = parse_candidate_verifier_verdict(json.dumps(missing))
        self.assertFalse(verdict["valid"])
        self.assertEqual(verdict["rejection_type"], "invalid_json")

    def test_batched_meta_routing_fails_closed_on_missing_and_duplicate_ids(self):
        payload = {
            "results": [
                _multidimensional_result("p0_c0"),
                _multidimensional_result("p1_c0"),
                _multidimensional_result("p1_c0"),
                _multidimensional_result("unknown"),
            ]
        }
        routed, diagnostics = parse_batched_candidate_verifier_payloads(
            json.dumps(payload),
            expected_candidate_ids=["p0_c0", "p1_c0", "p2_c0"],
        )
        self.assertFalse(parse_candidate_verifier_verdict(routed["p0_c0"])["valid"])
        self.assertFalse(parse_candidate_verifier_verdict(routed["p1_c0"])["valid"])
        self.assertFalse(parse_candidate_verifier_verdict(routed["p2_c0"])["valid"])
        self.assertEqual(diagnostics["duplicate_candidate_ids"], ["p1_c0"])
        self.assertEqual(diagnostics["missing_candidate_ids"], ["p2_c0"])
        self.assertEqual(diagnostics["unknown_candidate_ids"], ["unknown"])

    def test_batched_meta_routing_accepts_top_level_results_array(self):
        payload = [
            _multidimensional_result("p2_c0"),
            _multidimensional_result("p5_c1"),
        ]

        routed, diagnostics = parse_batched_candidate_verifier_payloads(
            json.dumps(payload),
            expected_candidate_ids=["p2_c0", "p5_c1"],
        )

        self.assertTrue(diagnostics["parse_valid"])
        self.assertEqual(diagnostics["returned_count"], 2)
        self.assertEqual(diagnostics["missing_candidate_ids"], [])
        self.assertTrue(parse_candidate_verifier_verdict(routed["p2_c0"])["valid"])
        self.assertTrue(parse_candidate_verifier_verdict(routed["p5_c1"])["valid"])

    def test_parse_candidate_verifier_verdict_requires_all_checks(self):
        verdict = parse_candidate_verifier_verdict(
            _verifier_payload(
                valid=True,
                parent_anchored=False,
                reason="Candidate is about the sample but not the parent.",
            )
        )
        self.assertFalse(verdict["valid"])
        self.assertEqual(verdict["rejection_type"], "parent_drift")

        accepted = parse_candidate_verifier_verdict(_verifier_payload())
        self.assertTrue(accepted["valid"])

    def test_parse_evorubric_meta_verifier_verdict(self):
        accepted = parse_candidate_verifier_verdict(
            json.dumps({"verdict": "VALID", "reason": "targeted and judgeable"})
        )
        self.assertTrue(accepted["valid"])

        rejected = parse_candidate_verifier_verdict(
            json.dumps(
                {
                    "verdict": "INVALID: Semantic Duplication",
                    "reason": "covered by an existing criterion",
                }
            )
        )
        self.assertFalse(rejected["valid"])
        self.assertEqual(rejected["rejection_type"], "redundant")

    def test_parse_collision_verifier_verdict_fails_closed_on_conflict(self):
        verdict = parse_collision_verifier_verdict(
            json.dumps(
                {
                    "comparisons": [
                        {
                            "criterion_id": "hydration",
                            "relation": "sibling_dimension",
                            "conflict": True,
                            "reason": "Candidate moved from folic acid to hydration.",
                        }
                    ],
                    "conflict_ids": ["hydration"],
                    "no_conflict": False,
                    "valid": False,
                    "reason": "cross-criterion leakage",
                }
            )
        )
        self.assertFalse(verdict["valid"])
        self.assertEqual(verdict["conflict_ids"], ["hydration"])

    def test_collision_pair_verdict_can_clear_false_positive(self):
        verdict = parse_collision_pair_verdict(
            json.dumps(
                {
                    "candidate_matches_existing": False,
                    "drifted_to_existing_dimension": False,
                    "duplicate": False,
                    "relation": "unrelated",
                    "conflict": False,
                    "valid": True,
                    "reason": "Literal criteria check different behaviors.",
                }
            )
        )
        self.assertTrue(verdict["valid"])
        self.assertFalse(verdict["conflict"])

    def test_collision_pair_verdict_cannot_clear_contradictory_drift(self):
        verdict = parse_collision_pair_verdict(
            json.dumps(
                {
                    "candidate_matches_existing": False,
                    "drifted_to_existing_dimension": True,
                    "duplicate": False,
                    "relation": "sibling_dimension",
                    "conflict": False,
                    "valid": True,
                    "reason": "Contradictory 4B fields must fail closed.",
                }
            )
        )
        self.assertTrue(verdict["conflict"])

    def test_negative_polarity_verdict_rejects_opposite_trigger(self):
        verdict = parse_negative_polarity_verdict(
            json.dumps(
                {
                    "parent_penalty_trigger": "penalize when calculating without inputs",
                    "candidate_penalty_trigger": "penalize when refusing to calculate",
                    "same_behavior_family": False,
                    "candidate_trigger_implies_parent_trigger": False,
                    "penalizes_opposite_or_avoidance": True,
                    "valid": False,
                    "reason": "Refusal is not calculation.",
                }
            )
        )
        self.assertFalse(verdict["valid"])

    def test_parent_novelty_rejects_refine_paraphrase_and_accepts_atomic_decompose(self):
        refine = parse_parent_novelty_verdict(
            json.dumps(
                {
                    "uncovered_constraints": [],
                    "proper_atomic_subset": False,
                    "is_paraphrase": True,
                    "reason": "Same constraints.",
                }
            ),
            source="refine",
        )
        self.assertFalse(refine["valid"])

        decompose = parse_parent_novelty_verdict(
            json.dumps(
                {
                    "uncovered_constraints": [],
                    "proper_atomic_subset": True,
                    "is_paraphrase": False,
                    "reason": "One component of broader history.",
                }
            ),
            source="decompose",
        )
        self.assertTrue(decompose["valid"])

        explicit_restatement = parse_parent_novelty_verdict(
            json.dumps(
                {
                    "uncovered_constraints": ["explicit mention of supplement use"],
                    "proper_atomic_subset": False,
                    "is_paraphrase": False,
                    "reason": "Claims explicitness is new.",
                }
            ),
            source="refine",
            parent_text="Suggests a folic acid supplement in addition to dietary folate.",
        )
        self.assertFalse(explicit_restatement["valid"])
        self.assertEqual(explicit_restatement["substantive_uncovered_constraints"], [])

        hallucinated_uncovered = parse_parent_novelty_verdict(
            json.dumps(
                {
                    "uncovered_constraints": ["long-term hydration monitoring"],
                    "proper_atomic_subset": False,
                    "is_paraphrase": False,
                    "reason": "Invented verifier explanation.",
                }
            ),
            source="refine",
            parent_text="Suggests a folic acid supplement.",
            candidate_text="Explicitly suggests a folic acid supplement.",
        )
        self.assertFalse(hallucinated_uncovered["valid"])

    def test_proposal_evidence_must_match_quotes_and_empirical_sides(self):
        candidate = {
            "positive_evidence": [{"rollout_id": 1, "quote": "asks about contacts"}],
            "negative_evidence": [{"rollout_id": 2, "reason": "absent"}],
        }
        valid = validate_proposal_evidence(
            candidate,
            ["The response asks about contacts.", "No exposure history."],
            [1, 0],
        )
        self.assertTrue(valid["valid"])

        invalid = validate_proposal_evidence(
            candidate,
            ["The response asks about contacts.", "No exposure history."],
            [0, 1],
        )
        self.assertFalse(invalid["valid"])

    def test_unsupported_numeric_claims_must_come_from_question_or_parent(self):
        self.assertEqual(
            find_unsupported_numeric_claims(
                "Requires at least 800 mcg daily.",
                "Parent requires at least 600 mcg daily.",
            ),
            ["800"],
        )
        self.assertEqual(
            find_unsupported_numeric_claims(
                "Requires at least 600 mcg daily.",
                "Parent requires at least 600 mcg daily.",
            ),
            [],
        )

    def test_apply_active_rubric_retains_parent_anchor_with_children(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
        )
        evolver = RubricEvolver(config)

        parent = [
            {
                "title": "Exposure",
                "description": "Takes a focused exposure history.",
                "weight": 5.0,
            },
            {
                "title": "Safety",
                "description": "Mentions when to seek urgent care.",
                "weight": 3.0,
            },
        ]
        prompt_text = "Patient with chronic cough."
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "prompt_text": np.array([prompt_text], dtype=object),
                "rubric": np.array([parent], dtype=object),
                "rubric_list": np.array([None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        prompt_key = str(batch.non_tensor_batch["rubric_evolution_prompt_key"][0])
        state = evolver.store.get(prompt_key)
        assert state is not None
        from verl.trainer.rubric_evolution import EvolvedCriterion

        state.candidates = [
            EvolvedCriterion(
                criterion_id="child_1",
                parent_index=0,
                parent_criterion_id="parent_0",
                title="TB contacts",
                description="Asks about close TB contacts.",
                weight=2.0,
                source="decompose",
                status="active",
                proposed_step=1,
                verify_history=[{"passed": True}],
            )
        ]

        evolver.apply_active_rubric_to_batch(batch)
        active = batch.non_tensor_batch["rubric"][0]
        self.assertEqual(len(active), 3)
        self.assertEqual(active[0]["description"], "Takes a focused exposure history.")
        self.assertEqual(active[1]["description"], "Asks about close TB contacts.")
        self.assertEqual(active[2]["description"], "Mentions when to seek urgent care.")

    def test_weight_transfer_zeros_parent_and_splits_full_signed_weight(self):
        config = OPSDConfig(
            rubric_evolution_child_weight_ratio=0.2,
            rubric_evolution_transfer_parent_weight_to_children=True,
        )
        evolver = RubricEvolver(config)
        parent = [
            {"title": "Positive", "description": "Positive parent.", "weight": 6.0},
            {"title": "Penalty", "description": "Negative parent.", "weight": -4.0},
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "prompt_text": np.array(["Question"], dtype=object),
                "rubric": np.array([parent], dtype=object),
                "rubric_list": np.array([None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        prompt_key = str(batch.non_tensor_batch["rubric_evolution_prompt_key"][0])
        state = evolver.store.get(prompt_key)
        assert state is not None
        state.candidates.extend(
            [
                EvolvedCriterion(
                    criterion_id="positive_child_1",
                    parent_index=0,
                    parent_criterion_id="parent_0",
                    title="Positive child 1",
                    description="First positive child.",
                    weight=0.0,
                    source="decompose",
                    status="active",
                ),
                EvolvedCriterion(
                    criterion_id="positive_child_2",
                    parent_index=0,
                    parent_criterion_id="parent_0",
                    title="Positive child 2",
                    description="Second positive child.",
                    weight=0.0,
                    source="decompose",
                    status="active",
                ),
                EvolvedCriterion(
                    criterion_id="negative_child",
                    parent_index=1,
                    parent_criterion_id="parent_1",
                    title="Negative child",
                    description="Negative child.",
                    weight=0.0,
                    source="decompose",
                    status="active",
                ),
            ]
        )
        # The configured 0.2 auxiliary ratio is intentionally ignored in
        # transfer mode: the entire original parent weight is conserved.
        state.rebalance_active_child_weights(0, config.rubric_evolution_child_weight_ratio)
        state.rebalance_active_child_weights(1, config.rubric_evolution_child_weight_ratio)
        active = state.build_active_rubric()

        self.assertEqual([item["weight"] for item in active], [0.0, 3.0, 3.0, 0.0, -4.0])
        self.assertEqual([item["weight"] for item in state.parent_rubric], [6.0, -4.0])

    def test_pending_candidate_activates_only_after_configured_delay(self):
        config = OPSDConfig(rubric_evolution_delayed_steps=2)
        evolver = RubricEvolver(config)
        parent = [{"title": "A", "description": "Parent A", "weight": 5.0}]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "prompt_text": np.array(["Question"], dtype=object),
                "rubric": np.array([parent], dtype=object),
                "rubric_list": np.array([None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        prompt_key = str(batch.non_tensor_batch["rubric_evolution_prompt_key"][0])
        state = evolver.store.get(prompt_key)
        assert state is not None
        state.candidates.append(
            EvolvedCriterion(
                criterion_id="child_pending",
                parent_index=0,
                parent_criterion_id="parent_0",
                title="Child",
                description="Child A",
                weight=1.0,
                source="decompose",
                status="pending",
                proposed_step=1,
                verify_history=[{"passed": True}],
            )
        )
        self.assertEqual(evolver.activate_pending(global_step=2)["rubric_evolution/activated_candidates"], 0.0)
        self.assertEqual(evolver.activate_pending(global_step=3)["rubric_evolution/activated_candidates"], 1.0)
        self.assertEqual(len(state.active_children()), 1)

    def test_verified_proposal_enters_only_a_future_reward_batch(self):
        config = OPSDConfig(
            rubric_evolution_delayed_steps=2,
            rubric_evolution_child_weight_ratio=0.2,
        )

        def judge_fn(inp):
            score = 1.0 if "contacts" in inp["response"] else 0.0
            return {"overall": score, "criterion_0_score": score, "judge_error": 0.0}

        evolver = RubricEvolver(config, judge_fn=judge_fn)
        parent = [{"title": "Exposure", "description": "Takes an exposure history.", "weight": 5.0}]
        current_batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(current_batch)
        evolver.observe(
            current_batch,
            {"criterion_0_score": [0.0, 0.0]},
            global_step=1,
        )
        target = evolver.select_targets(global_step=1)[0]
        target.group_responses = ["Asks about contacts.", "No exposure questions."]
        proposal = json.dumps(
            {
                "subcriteria": [
                    {
                        "title": "Contact exposure",
                        "description": "Asks about relevant contacts.",
                    }
                ]
            }
        )
        metrics = evolver.evolve_targets(
            [target],
            global_step=1,
            contrast_responses={},
            proposal_payloads={(target.prompt_key, 0): proposal},
            proposal_verifier_payloads={
                (target.prompt_key, 0, 0): json.dumps(_multidimensional_result())
            },
            activate_immediately=False,
        )

        self.assertEqual(metrics["rubric_evolution/accepted_candidates"], 1.0)
        state = evolver.store.get(target.prompt_key)
        assert state is not None
        self.assertEqual(len(state.pending_children()), 1)
        # The proposal step's already-scored batch still has only the parent.
        self.assertEqual(len(current_batch.non_tensor_batch["rubric"][0]), 1)
        self.assertEqual(evolver.activate_pending(global_step=2)["rubric_evolution/activated_candidates"], 0.0)
        self.assertEqual(evolver.activate_pending(global_step=3)["rubric_evolution/activated_candidates"], 1.0)

        future_batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "prompt_text": np.array(["Question"], dtype=object),
                "rubric": np.array([parent], dtype=object),
                "rubric_list": np.array([None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(future_batch)
        future_reward_rubric = future_batch.non_tensor_batch["rubric"][0]
        self.assertEqual(len(future_reward_rubric), 2)
        self.assertEqual(future_reward_rubric[1]["criterion_id"], state.candidates[0].criterion_id)
        self.assertEqual(future_reward_rubric[1]["weight"], 1.0)

    def test_store_checkpoint_round_trip_preserves_reward_rubric_state(self):
        config = OPSDConfig(rubric_evolution_child_weight_ratio=0.2)
        evolver = RubricEvolver(config)
        parent = [
            {"title": "A", "description": "Parent A", "weight": 5.0},
            {"title": "B", "description": "Parent B", "weight": -3.0},
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        evolver.observe(
            batch,
            {
                "criterion_0_score": [0.0, 0.0],
                "criterion_1_score": [1.0, 1.0],
            },
            global_step=4,
        )
        prompt_key = str(batch.non_tensor_batch["rubric_evolution_prompt_key"][0])
        state = evolver.store.get(prompt_key)
        assert state is not None
        state.candidates.extend(
            [
                EvolvedCriterion(
                    criterion_id="active_child",
                    parent_index=0,
                    parent_criterion_id="parent_0",
                    title="Active",
                    description="Active reward criterion.",
                    weight=1.0,
                    source="decompose",
                    status="active",
                    proposed_step=2,
                    verify_history=[{"passed": True, "score_vector": [0.0, 1.0]}],
                ),
                EvolvedCriterion(
                    criterion_id="pending_child",
                    parent_index=1,
                    parent_criterion_id="parent_1",
                    title="Pending",
                    description="Pending reward criterion.",
                    weight=-0.6,
                    source="decompose",
                    status="pending",
                    proposed_step=4,
                    verify_history=[{"passed": True, "score_vector": [1.0, 0.0]}],
                ),
            ]
        )
        state.criterion_meta[0].active_child_ids = ["active_child"]
        state.rebalance_active_child_weights(0, config.rubric_evolution_child_weight_ratio)

        serialized = json.loads(json.dumps(evolver.store.state_dict()))
        restored = RubricEvolver(config)
        restored.store.load_state_dict(serialized)
        restored_state = restored.store.get(prompt_key)
        assert restored_state is not None

        self.assertEqual(
            [candidate.status for candidate in restored_state.candidates],
            ["active", "pending"],
        )
        self.assertEqual(
            restored_state.criterion_meta[0].active_child_ids,
            ["active_child"],
        )
        self.assertEqual(
            restored_state.candidates[1].verify_history[0]["score_vector"],
            [1.0, 0.0],
        )
        active_rubric = restored_state.build_active_rubric()
        self.assertEqual(
            [item["criterion_id"] for item in active_rubric],
            ["parent_0", "active_child", "parent_1"],
        )
        self.assertEqual(active_rubric[1]["weight"], 1.0)

    def test_observe_updates_dead_and_mixed_counts(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
        )
        evolver = RubricEvolver(config)
        parent = [
            {"title": "A", "description": "Criterion A", "weight": 1.0},
            {"title": "B", "description": "Criterion B", "weight": 1.0},
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        metrics = evolver.observe(
            batch,
            {
                "criterion_0_score": [0.0, 0.0],
                "criterion_1_score": [1.0, 0.0],
            },
            global_step=1,
        )
        self.assertEqual(metrics["rubric_evolution/dead_count"], 1.0)
        self.assertEqual(metrics["rubric_evolution/mixed_count"], 1.0)

    def test_observe_does_not_classify_incomplete_judge_group_as_dead(self):
        evolver = RubricEvolver(OPSDConfig())
        parent = [{"title": "A", "description": "Criterion A", "weight": 1.0}]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        metrics = evolver.observe(
            batch,
            {"criterion_0_score": [1.0]},
            global_step=1,
        )
        self.assertEqual(metrics["rubric_evolution/incomplete_score_groups"], 1.0)
        self.assertEqual(metrics["rubric_evolution/criteria_observed"], 0.0)
        self.assertEqual(evolver.select_targets(global_step=1), [])

    def test_select_targets_includes_all_dead_and_saturated_criteria(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
        )
        evolver = RubricEvolver(config)
        parent = [
            {"title": "A", "description": "Criterion A", "weight": 1.0},
            {"title": "B", "description": "Criterion B", "weight": 1.0},
            {"title": "C", "description": "Criterion C", "weight": 1.0},
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        evolver.observe(
            batch,
            {
                "criterion_0_score": [0.0, 0.0],
                "criterion_1_score": [0.0, 0.0],
                "criterion_2_score": [1.0, 1.0],
            },
            global_step=1,
        )
        targets = evolver.select_targets(global_step=1)
        self.assertEqual(len(targets), 3)
        self.assertEqual({target.parent_index for target in targets}, {0, 1, 2})

    def test_select_targets_can_only_include_example_criteria(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
            rubric_evolution_only_example_criteria=True,
        )
        evolver = RubricEvolver(config)
        parent = [
            {"title": "communication_quality (cluster)", "description": "Broad quality.", "weight": 5.0},
            {"title": "accuracy (example)", "description": "Specific fact.", "weight": 5.0},
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        evolver.observe(
            batch,
            {
                "criterion_0_score": [0.0, 0.0],
                "criterion_1_score": [0.0, 0.0],
            },
            global_step=1,
        )
        targets = evolver.select_targets(global_step=1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].parent_index, 1)

    def test_select_targets_skips_existing_children_and_caps_per_step(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
            rubric_evolution_max_targets_per_step=2,
            rubric_evolution_retry_cooldown_steps=0,
        )
        evolver = RubricEvolver(config)
        parent = [
            {"title": "A", "description": "Criterion A", "weight": 1.0},
            {"title": "B", "description": "Criterion B", "weight": 1.0},
            {"title": "C", "description": "Criterion C", "weight": 1.0},
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        evolver.observe(
            batch,
            {
                "criterion_0_score": [0.0, 0.0],
                "criterion_1_score": [0.0, 0.0],
                "criterion_2_score": [0.0, 0.0],
            },
            global_step=1,
        )
        targets = evolver.select_targets(global_step=1)
        self.assertEqual(len(targets), 2)

        prompt_key = targets[0].prompt_key
        state = evolver.store.get(prompt_key)
        assert state is not None
        state.candidates.append(
            EvolvedCriterion(
                criterion_id="child_1",
                parent_index=0,
                parent_criterion_id="parent_0",
                title="A child",
                description="A child criterion.",
                weight=1.0,
                source="decompose",
                status=CANDIDATE_ACTIVE,
                proposed_step=1,
                verify_history=[{"passed": True}],
            )
        )
        targets = evolver.select_targets(global_step=2)
        self.assertNotIn(0, {target.parent_index for target in targets})

    def test_observe_clears_stale_group_cache_between_batches(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
            rubric_evolution_max_targets_per_step=0,
        )
        evolver = RubricEvolver(config)
        parent = [{"title": "A", "description": "Criterion A", "weight": 1.0}]

        first_batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["First prompt", "First prompt"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(first_batch)
        evolver.observe(first_batch, {"criterion_0_score": [0.0, 0.0]}, global_step=1)
        self.assertEqual(len(evolver.select_targets(global_step=1)), 1)

        second_batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g2", "g2"], dtype=object),
                "prompt_text": np.array(["Second prompt", "Second prompt"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(second_batch)
        evolver.observe(second_batch, {"criterion_0_score": [1.0, 1.0]}, global_step=2)
        targets = evolver.select_targets(global_step=2)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].prompt_text, "Second prompt")
        self.assertEqual(targets[0].group_indices, [0, 1])

    def test_observe_aligns_parent_scores_after_replacement(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
        )
        evolver = RubricEvolver(config)
        parent = [
            {"title": "A", "description": "Criterion A", "weight": 1.0},
            {"title": "B", "description": "Criterion B", "weight": 1.0},
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        prompt_key = str(batch.non_tensor_batch["rubric_evolution_prompt_key"][0])
        state = evolver.store.get(prompt_key)
        assert state is not None
        state.candidates.append(
            EvolvedCriterion(
                criterion_id="child_1",
                parent_index=0,
                parent_criterion_id="parent_0",
                title="A child",
                description="A child criterion.",
                weight=1.0,
                source="decompose",
                status=CANDIDATE_ACTIVE,
                proposed_step=1,
                verify_history=[{"passed": True}],
            )
        )

        evolver.apply_active_rubric_to_batch(batch)
        # Active rubric is [parent A, child(A), parent B]. Parent B must read
        # criterion_2, not either of the A criteria.
        metrics = evolver.observe(
            batch,
            {
                "criterion_0_score": [0.0, 0.0],
                "criterion_1_score": [0.0, 0.0],
                "criterion_2_score": [1.0, 1.0],
            },
            global_step=2,
        )

        self.assertEqual(metrics["rubric_evolution/saturated_count"], 1.0)
        self.assertEqual(state.criterion_meta[1].pass_rate, 1.0)

    def test_evolve_targets_can_activate_immediately_for_current_batch(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
        )

        def judge_fn(inp):
            response = inp["response"]
            score = 1.0 if "contacts" in response else 0.0
            return {"overall": score, "criterion_0_score": score}

        evolver = RubricEvolver(config, judge_fn=judge_fn)
        parent = [
            {
                "title": "Exposure",
                "description": "Takes a focused exposure history.",
                "weight": 5.0,
            }
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        evolver.observe(batch, {"criterion_0_score": [0.0, 0.0]}, global_step=1)
        targets = evolver.select_targets(global_step=1)
        self.assertEqual(len(targets), 1)
        targets[0].group_responses = ["Asks about close contacts.", "Still missing exposure."]

        prompt_key = targets[0].prompt_key
        metrics = evolver.evolve_targets(
            targets,
            global_step=1,
            contrast_responses={prompt_key: "Asks about close TB contacts."},
            proposal_payloads={
                (prompt_key, 0): json.dumps(
                    {
                        "subcriteria": [
                            {
                                "title": "TB contacts",
                                "description": "Asks about close TB contacts.",
                                "positive_evidence": [
                                    {"rollout_id": 1, "quote": "Asks about close contacts."}
                                ],
                                "negative_evidence": [
                                    {"rollout_id": 2, "reason": "Target behavior is absent."}
                                ],
                            }
                        ]
                    }
                )
            },
            proposal_verifier_payloads={(prompt_key, 0, 0): _verifier_payload()},
            activate_immediately=True,
        )

        self.assertEqual(metrics["rubric_evolution/accepted_candidates"], 1.0)
        self.assertEqual(metrics["rubric_evolution/verifier_candidates"], 1.0)
        state = evolver.store.get(prompt_key)
        assert state is not None
        self.assertEqual(len(state.active_children()), 1)
        self.assertEqual(state.active_children()[0].status, CANDIDATE_ACTIVE)
        json.dumps(state.active_children()[0].verify_history)

        evolver.apply_active_rubric_to_batch(batch)
        active_rubric = batch.non_tensor_batch["rubric"][0]
        self.assertEqual(len(active_rubric), 2)
        self.assertEqual(active_rubric[0]["description"], "Takes a focused exposure history.")
        self.assertEqual(active_rubric[1]["description"], "Asks about close TB contacts.")

    def test_synthetic_contrast_cannot_rescue_zero_signal_candidate(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
        )

        def judge_fn(inp):
            description = inp["rubric"][0]["description"]
            response = inp["response"]
            if "focused exposure history" in description:
                return {"overall": 0.0, "criterion_0_score": 0.0}
            score = 1.0 if "contacts" in response else 0.0
            return {"overall": score, "criterion_0_score": score}

        evolver = RubricEvolver(config, judge_fn=judge_fn)
        parent = [
            {
                "title": "Exposure",
                "description": "Takes a focused exposure history.",
                "weight": 5.0,
            }
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        evolver.observe(batch, {"criterion_0_score": [0.0, 0.0]}, global_step=1)
        targets = evolver.select_targets(global_step=1)
        self.assertEqual(len(targets), 1)
        targets[0].group_responses = ["No exposure details.", "Still missing exposure."]

        prompt_key = targets[0].prompt_key
        metrics = evolver.evolve_targets(
            targets,
            global_step=1,
            contrast_responses={prompt_key: "Asks about close TB contacts."},
            proposal_payloads={
                (prompt_key, 0): json.dumps(
                    {
                        "subcriteria": [
                            {
                                "title": "TB contacts",
                                "description": "Asks about close TB contacts.",
                                "positive_evidence": [
                                    {"rollout_id": 1, "quote": "Asks about contacts."}
                                ],
                                "negative_evidence": [
                                    {"rollout_id": 2, "reason": "Target behavior is absent."}
                                ],
                            }
                        ]
                    }
                )
            },
            activate_immediately=True,
        )

        self.assertEqual(metrics["rubric_evolution/dead_contrast_parent_failed"], 0.0)
        self.assertEqual(metrics["rubric_evolution/accepted_candidates"], 0.0)
        self.assertEqual(metrics["rubric_evolution/verify_failed_no_signal"], 1.0)
        state = evolver.store.get(prompt_key)
        assert state is not None
        self.assertEqual(len(state.active_children()), 0)

    def test_empirical_verifier_rejects_singleton_and_accepts_supported_split(self):
        config = OPSDConfig()

        def judge_fn(inp):
            score = 1.0 if "target evidence" in inp["response"] else 0.0
            return {"overall": score, "criterion_0_score": score, "judge_error": 0.0}

        evolver = RubricEvolver(config, judge_fn=judge_fn)
        passed, info = evolver._verify_candidate(
            prompt_text="Question",
            parent_item={"title": "Parent", "description": "Broad behavior", "weight": 1.0},
            candidate={"title": "Child", "description": "Target behavior", "weight": 1.0},
            source="decompose",
            rollout_responses=["absent"] * 7 + ["target evidence appears here"],
        )

        self.assertFalse(passed)
        self.assertEqual(info["response_count"], 8)
        self.assertEqual(info["hit_count"], 1)
        self.assertEqual(info["hit_vector"], [0, 0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(info["required_split_support"], 2)

        passed, info = evolver._verify_candidate(
            prompt_text="Question",
            parent_item={"title": "Parent", "description": "Broad behavior", "weight": 1.0},
            candidate={"title": "Child", "description": "Target behavior", "weight": 1.0},
            source="decompose",
            rollout_responses=["absent"] * 6 + ["target evidence one", "target evidence two"],
        )
        self.assertTrue(passed)
        self.assertEqual(info["hit_count"], 2)

    def test_empirical_verifier_preserves_empty_member_and_rejects_incomplete_group(self):
        config = OPSDConfig()

        def judge_fn(inp):
            score = 1.0 if "target" in inp["response"] else 0.0
            return {"criterion_0_score": score, "judge_error": 0.0}

        evolver = RubricEvolver(config, judge_fn=judge_fn)
        common = {
            "prompt_text": "Question",
            "parent_item": {"title": "Parent", "description": "Broad", "weight": 1.0},
            "candidate": {"title": "Child", "description": "Target", "weight": 1.0},
            "source": "decompose",
        }
        passed, info = evolver._verify_candidate(
            **common,
            rollout_responses=["", "target appears"],
            expected_response_count=2,
        )
        self.assertTrue(passed)
        self.assertEqual(info["response_count"], 2)

        passed, info = evolver._verify_candidate(
            **common,
            rollout_responses=["absent"] * 7,
            expected_response_count=8,
        )
        self.assertFalse(passed)
        self.assertEqual(info["reason"], "incomplete_rollout_group")

    def test_empirical_verifier_fails_closed_on_judge_error(self):
        config = OPSDConfig()

        def judge_fn(inp):
            if "broken" in inp["response"]:
                return {"criterion_0_score": 1.0, "judge_error": 1.0}
            return {"criterion_0_score": 0.0, "judge_error": 0.0}

        evolver = RubricEvolver(config, judge_fn=judge_fn)
        passed, info = evolver._verify_candidate(
            prompt_text="Question",
            parent_item={"title": "Parent", "description": "Broad behavior", "weight": 1.0},
            candidate={"title": "Child", "description": "Target behavior", "weight": 1.0},
            source="refine",
            rollout_responses=["normal", "broken"],
        )

        self.assertFalse(passed)
        self.assertEqual(info["reason"], "judge_error")

    def test_decompose_rejects_sample_level_drift_from_parent(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
        )

        def judge_fn(inp):
            description = inp["rubric"][0]["description"]
            response = inp["response"]
            if "geographic location" in description:
                score = 1.0 if "where do you live" in response.lower() else 0.0
            else:
                score = 1.0 if "feeding tube" in response.lower() else 0.0
            return {"overall": score, "criterion_0_score": score}

        evolver = RubricEvolver(config, judge_fn=judge_fn)
        parent = [
            {
                "title": "Local resources",
                "description": (
                    "Asks about the user's geographic location to provide "
                    "region-specific resources relevant to dysphagia assessment."
                ),
                "weight": 3.0,
            }
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(
                    ["My father may need a feeding tube.", "My father may need a feeding tube."],
                    dtype=object,
                ),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        evolver.observe(batch, {"criterion_0_score": [0.0, 0.0]}, global_step=1)
        targets = evolver.select_targets(global_step=1)
        self.assertEqual(len(targets), 1)
        targets[0].group_responses = ["Feeding tube is a last resort.", "No location asked."]
        prompt_key = targets[0].prompt_key

        metrics = evolver.evolve_targets(
            targets,
            global_step=1,
            contrast_responses={prompt_key: "A feeding tube is usually a last-resort intervention."},
            proposal_payloads={
                (prompt_key, 0): json.dumps(
                    {
                        "subcriteria": [
                            {
                                "title": "Feeding tube last resort",
                                "description": "Clarifies feeding tube as a last-resort intervention.",
                            }
                        ]
                    }
                )
            },
            proposal_verifier_payloads={
                (prompt_key, 0, 0): _verifier_payload(
                    valid=False,
                    parent_anchored=False,
                    reason="Candidate is relevant to the question but not inherited from the location parent.",
                )
            },
            activate_immediately=True,
        )

        self.assertEqual(metrics["rubric_evolution/accepted_candidates"], 0.0)
        self.assertEqual(metrics["rubric_evolution/verify_failed_parent_drift"], 1.0)
        examples = evolver.pop_recent_evolution_examples()
        self.assertEqual(examples[-1]["outcome"], "rejected")
        self.assertIn("verifier_parent_drift", examples[-1]["verify_detail"])

    def test_refine_rejects_sample_level_drift_from_parent(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
        )

        def judge_fn(inp):
            description = inp["rubric"][0]["description"]
            response = inp["response"]
            if "geographic location" in description:
                return {"overall": 1.0, "criterion_0_score": 1.0}
            score = 1.0 if "aspiration" in response.lower() else 0.0
            return {"overall": score, "criterion_0_score": score}

        evolver = RubricEvolver(config, judge_fn=judge_fn)
        parent = [
            {
                "title": "Local resources",
                "description": (
                    "Asks about the user's geographic location to provide "
                    "region-specific resources relevant to dysphagia assessment."
                ),
                "weight": 3.0,
            }
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(
                    ["My father may need a feeding tube.", "My father may need a feeding tube."],
                    dtype=object,
                ),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        evolver.observe(batch, {"criterion_0_score": [1.0, 1.0]}, global_step=1)
        targets = evolver.select_targets(global_step=1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].state, CRITERION_STATE_SATURATED)
        targets[0].group_responses = ["Mentions aspiration risk.", "No aspiration detail."]
        prompt_key = targets[0].prompt_key

        metrics = evolver.evolve_targets(
            targets,
            global_step=1,
            contrast_responses={prompt_key: "Aspiration risk is the main concern."},
            proposal_payloads={
                (prompt_key, 0): json.dumps(
                    {
                        "refinements": [
                            {
                                "title": "Aspiration risk",
                                "description": "Mentions aspiration risk as the primary medical concern.",
                            }
                        ]
                    }
                )
            },
            proposal_verifier_payloads={
                (prompt_key, 0, 0): _verifier_payload(
                    valid=False,
                    parent_anchored=False,
                    reason="Aspiration risk is a new sibling criterion, not a stricter location criterion.",
                )
            },
            activate_immediately=True,
        )

        self.assertEqual(metrics["rubric_evolution/refine_accepted"], 0.0)
        self.assertEqual(metrics["rubric_evolution/verify_failed_parent_drift"], 1.0)
        examples = evolver.pop_recent_evolution_examples()
        self.assertEqual(examples[-1]["outcome"], "rejected")
        self.assertIn("verifier_parent_drift", examples[-1]["verify_detail"])

    def test_policy_verifier_rejects_redundant_child(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
        )

        def judge_fn(inp):
            response = inp["response"]
            score = 1.0 if "exposure" in response else 0.0
            return {"overall": score, "criterion_0_score": score}

        evolver = RubricEvolver(config, judge_fn=judge_fn)
        parent = [
            {
                "title": "Exposure",
                "description": "Takes a focused exposure history.",
                "weight": 5.0,
            }
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        evolver.observe(batch, {"criterion_0_score": [0.0, 0.0]}, global_step=1)
        targets = evolver.select_targets(global_step=1)
        targets[0].group_responses = ["No details.", "Still missing."]
        prompt_key = targets[0].prompt_key

        metrics = evolver.evolve_targets(
            targets,
            global_step=1,
            contrast_responses={prompt_key: "Asks a focused exposure history."},
            proposal_payloads={
                (prompt_key, 0): json.dumps(
                    {
                        "subcriteria": [
                            {
                                "title": "Focused exposure history",
                                "description": "Takes a focused exposure history.",
                            }
                        ]
                    }
                )
            },
            proposal_verifier_payloads={
                (prompt_key, 0, 0): _verifier_payload(
                    valid=False,
                    not_redundant_with_parent=False,
                    reason="Candidate is a paraphrase of the parent.",
                )
            },
            activate_immediately=True,
        )

        self.assertEqual(metrics["rubric_evolution/accepted_candidates"], 0.0)
        self.assertEqual(metrics["rubric_evolution/verify_failed_redundant"], 1.0)

    def test_evolve_records_wandb_examples(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
        )

        def judge_fn(inp):
            response = inp["response"]
            score = 1.0 if "contacts" in response else 0.0
            return {"overall": score, "criterion_0_score": score}

        evolver = RubricEvolver(config, judge_fn=judge_fn)
        parent = [
            {
                "title": "Exposure",
                "description": "Takes a focused exposure history.",
                "weight": 5.0,
            }
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        evolver.observe(batch, {"criterion_0_score": [0.0, 0.0]}, global_step=1)
        targets = evolver.select_targets(global_step=1)
        targets[0].group_responses = ["Asks about contacts.", "Still missing exposure."]
        prompt_key = targets[0].prompt_key
        evolver.evolve_targets(
            targets,
            global_step=1,
            contrast_responses={prompt_key: "Asks about close TB contacts."},
            proposal_payloads={
                (prompt_key, 0): json.dumps(
                    {
                        "subcriteria": [
                            {
                                "title": "TB contacts",
                                "description": "Asks about close TB contacts.",
                                "positive_evidence": [
                                    {"rollout_id": 1, "quote": "Asks about contacts."}
                                ],
                                "negative_evidence": [
                                    {"rollout_id": 2, "reason": "Target behavior is absent."}
                                ],
                            }
                        ]
                    }
                )
            },
            activate_immediately=True,
        )
        examples = evolver.pop_recent_evolution_examples()
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["outcome"], "accepted")
        self.assertIn("Question", examples[0]["prompt"])
        self.assertIn("TB contacts", examples[0]["contrast"])
        self.assertEqual(examples[0]["candidate_status"], CANDIDATE_ACTIVE)
        self.assertEqual(examples[0]["active_rubric_before_count"], 1)
        self.assertIn("Takes a focused exposure history.", examples[0]["active_rubric_before"])
        self.assertNotIn("Asks about close TB contacts.", examples[0]["active_rubric_before"])
        self.assertIn("Asks about close TB contacts.", examples[0]["active_rubric_added"])
        self.assertEqual(examples[0]["active_rubric_replaced_parent"], "")
        self.assertIn("Asks about close TB contacts.", examples[0]["active_rubric_replacement_children"])
        self.assertEqual(examples[0]["active_rubric_after_count"], 2)
        self.assertIn("Takes a focused exposure history.", examples[0]["active_rubric_after"])
        self.assertIn("Asks about close TB contacts.", examples[0]["active_rubric_after"])

    def test_multiple_children_replace_parent_and_split_weight_budget(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
            rubric_evolution_child_weight_ratio=1.0,
        )

        def judge_fn(inp):
            description = inp["rubric"][0]["description"]
            response = inp["response"]
            if "contacts" in description:
                score = 1.0 if "contacts" in response else 0.0
            else:
                score = 1.0 if "travel" in response else 0.0
            return {"overall": score, "criterion_0_score": score}

        evolver = RubricEvolver(config, judge_fn=judge_fn)
        parent = [
            {
                "title": "Exposure",
                "description": "Takes a focused exposure history.",
                "weight": 6.0,
            }
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        evolver.observe(batch, {"criterion_0_score": [0.0, 0.0]}, global_step=1)
        targets = evolver.select_targets(global_step=1)
        targets[0].group_responses = ["Asks about contacts.", "Asks about travel."]
        prompt_key = targets[0].prompt_key

        metrics = evolver.evolve_targets(
            targets,
            global_step=1,
            contrast_responses={prompt_key: "Asks about contacts and travel."},
            proposal_payloads={
                (prompt_key, 0): json.dumps(
                    {
                        "subcriteria": [
                            {
                                "title": "TB contacts",
                                "description": "Asks about close TB contacts.",
                                "positive_evidence": [
                                    {"rollout_id": 1, "quote": "Asks about contacts."}
                                ],
                                "negative_evidence": [
                                    {"rollout_id": 2, "reason": "Target behavior is absent."}
                                ],
                            },
                            {
                                "title": "Travel exposure",
                                "description": "Asks about relevant travel exposure.",
                                "positive_evidence": [
                                    {"rollout_id": 2, "quote": "Asks about travel."}
                                ],
                                "negative_evidence": [
                                    {"rollout_id": 1, "reason": "Target behavior is absent."}
                                ],
                            },
                        ]
                    }
                )
            },
            activate_immediately=True,
        )

        self.assertEqual(metrics["rubric_evolution/accepted_candidates"], 2.0)
        state = evolver.store.get(prompt_key)
        assert state is not None
        children = state.active_children()
        self.assertEqual(len(children), 2)
        self.assertEqual([child.weight for child in children], [3.0, 3.0])
        evolver.apply_active_rubric_to_batch(batch)
        active_rubric = batch.non_tensor_batch["rubric"][0]
        self.assertEqual(len(active_rubric), 3)
        self.assertIn("Takes a focused exposure history.", [item["description"] for item in active_rubric])

    def test_negative_parent_evolves_negative_child_weight(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
            rubric_evolution_child_weight_ratio=0.5,
        )

        def judge_fn(inp):
            description = inp["rubric"][0]["description"]
            response = inp["response"]
            if "abruptly stopping" in description:
                score = 1.0 if "stop patch now" in response else 0.0
            else:
                score = 1.0 if "stop" in response else 0.0
            return {"overall": score, "criterion_0_score": score}

        evolver = RubricEvolver(config, judge_fn=judge_fn)
        parent = [
            {
                "title": "accuracy (example)",
                "description": "Advises abruptly stopping the birth control patch.",
                "weight": -4.0,
            }
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        evolver.observe(batch, {"criterion_0_score": [1.0, 1.0]}, global_step=1)
        targets = evolver.select_targets(global_step=1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].state, CRITERION_STATE_DEAD)
        targets[0].group_responses = ["stop patch now", "consult before changing it"]
        prompt_key = targets[0].prompt_key

        metrics = evolver.evolve_targets(
            targets,
            global_step=1,
            contrast_responses={prompt_key: "Talk to your clinician before changing the patch."},
            proposal_payloads={
                (prompt_key, 0): json.dumps(
                    {
                        "subcriteria": [
                            {
                                "title": "Abrupt patch discontinuation",
                                "description": "Advises abruptly stopping the patch.",
                                "positive_evidence": [
                                    {"rollout_id": 1, "quote": "stop patch now"}
                                ],
                                "negative_evidence": [
                                    {"rollout_id": 2, "reason": "No abrupt-stop advice."}
                                ],
                            }
                        ]
                    }
                )
            },
            activate_immediately=True,
        )

        self.assertEqual(metrics["rubric_evolution/accepted_candidates"], 1.0)
        state = evolver.store.get(prompt_key)
        assert state is not None
        self.assertEqual(state.active_children()[0].weight, -2.0)
        examples = evolver.pop_recent_evolution_examples()
        self.assertEqual(examples[0]["parent_polarity"], "negative")

    def test_negative_saturated_parent_is_skipped_when_refinement_cannot_split(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
        )

        def judge_fn(inp):
            description = inp["rubric"][0]["description"]
            response = inp["response"]
            if "Advises abruptly stopping" in description:
                return {"overall": 0.0, "criterion_0_score": 0.0}
            score = 1.0 if "no consult" in response else 0.0
            return {"overall": score, "criterion_0_score": score}

        evolver = RubricEvolver(config, judge_fn=judge_fn)
        parent = [
            {
                "title": "accuracy (example)",
                "description": "Advises abruptly stopping the birth control patch.",
                "weight": -4.0,
            }
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(["Question", "Question"], dtype=object),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        evolver.observe(batch, {"criterion_0_score": [0.0, 0.0]}, global_step=1)
        targets = evolver.select_targets(global_step=1)
        self.assertEqual(targets, [])

    def test_negative_parent_rejects_positive_requirement_child(self):
        config = OPSDConfig(
            enable_grpo_opsd_simple=True,
            enable_rubric_guided_rlsd=True,
            rubric_guided_teacher_template="teacher.jinja",
            enable_rubric_evolution=True,
            rubric_evolution_contrast_template="contrast.jinja",
            rubric_evolution_decompose_template="decompose.jinja",
            rubric_evolution_refine_template="refine.jinja",
        )

        def judge_fn(inp):
            description = inp["rubric"][0]["description"]
            response = inp["response"].lower()
            if "fails to identify" in description.lower():
                return {"overall": 0.0, "criterion_0_score": 0.0}
            score = 1.0 if "emergency services" in response else 0.0
            return {"overall": score, "criterion_0_score": score}

        evolver = RubricEvolver(config, judge_fn=judge_fn)
        parent = [
            {
                "title": "Emergency omission",
                "description": "Fails to identify that calling emergency services is mandatory.",
                "weight": -8.0,
            }
        ]
        batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "uid": np.array(["g1", "g1"], dtype=object),
                "prompt_text": np.array(
                    ["Chest pain and trouble breathing.", "Chest pain and trouble breathing."],
                    dtype=object,
                ),
                "rubric": np.array([parent, parent], dtype=object),
                "rubric_list": np.array([None, None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(batch)
        evolver.observe(batch, {"criterion_0_score": [1.0, 1.0]}, global_step=1)
        targets = evolver.select_targets(global_step=1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].state, CRITERION_STATE_DEAD)
        targets[0].group_responses = ["Call emergency services.", "No emergency services."]
        prompt_key = targets[0].prompt_key

        metrics = evolver.evolve_targets(
            targets,
            global_step=1,
            contrast_responses={prompt_key: "Call emergency services immediately."},
            proposal_payloads={
                (prompt_key, 0): json.dumps(
                    {
                        "subcriteria": [
                            {
                                "title": "Emergency referral prioritization",
                                "description": (
                                    "The response must explicitly state that calling "
                                    "emergency services is the first and non-negotiable action."
                                ),
                            }
                        ]
                    }
                )
            },
            proposal_verifier_payloads={
                (prompt_key, 0, 0): _verifier_payload(
                    valid=False,
                    polarity_preserving=False,
                    reason="Negative parent child is phrased as a positive requirement.",
                )
            },
            activate_immediately=True,
        )

        self.assertEqual(metrics["rubric_evolution/decompose_accepted"], 0.0)
        self.assertEqual(metrics["rubric_evolution/verify_failed_polarity_mismatch"], 1.0)
        examples = evolver.pop_recent_evolution_examples()
        self.assertEqual(examples[-1]["outcome"], "rejected")
        self.assertIn("verifier_polarity_mismatch", examples[-1]["verify_detail"])


if __name__ == "__main__":
    unittest.main()
