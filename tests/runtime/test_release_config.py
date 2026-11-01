"""Validate the launcher against the actual distributed trainer configuration."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from omegaconf import OmegaConf

from liverubric.launch import DEFAULT_CONFIG, resolved_config
from verl.trainer.config import PPOConfig
from verl.trainer.opsd_config import OPSDConfig
from verl.trainer.rubric_evolution import RubricEvolver


class ReleaseRuntimeTests(unittest.TestCase):
    def test_complete_configuration_matches_real_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(config=DEFAULT_CONFIG, model=root / "model", val_file=None,
                output=root / "run", gpus=4)
            raw = resolved_config(args, root / "initialization")
            config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(PPOConfig()), raw))
            config.deep_post_init()
            self.assertEqual(config.worker.actor.kl_coef, .001)
            self.assertEqual(config.worker.rollout.response_length, 4096)
            self.assertFalse(config.opsd.enable_rubric_initialization)

    def test_real_empirical_gate_requires_two_each_side(self):
        evolver = RubricEvolver(OPSDConfig(rubric_evolution_min_split_count=2))
        for hits in range(9):
            passed, info = evolver.verify_candidate_on_group(
                prompt_text="Q", candidate=dict(title="Safety", description="Mentions safety", weight=1),
                source="gap_expand", rollout_responses=[f"Answer {i}" for i in range(8)],
                expected_response_count=8,
                precomputed_scores=[{"criterion_0_score": int(i < hits)} for i in range(8)])
            self.assertEqual(passed, 2 <= hits <= 6)
            self.assertEqual(info["required_split_support"], 2)


if __name__ == "__main__":
    unittest.main()
