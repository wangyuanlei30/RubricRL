import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from verl.protocol import DataProto
from verl.trainer.opsd_config import OPSDConfig
from verl.trainer.rubric_evolution import EvolvedCriterion, RubricEvolver


MODULE_PATH = (
    Path(__file__).parents[2]
    / "rewards"
    / "rubric_llm_judge.py"
)
SPEC = importlib.util.spec_from_file_location("rubric_llm_judge_under_test", MODULE_PATH)
judge = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(judge)


class RubricLLMJudgeTest(unittest.TestCase):
    def setUp(self):
        judge._CACHE.clear()

    def test_healthbench_weighting_adds_negative_points_for_bad_criteria(self):
        rubric = [
            {"description": "Includes advice.", "weight": 10},
            {"description": "Fails to mention warning signs.", "weight": -4},
        ]

        score = judge._compute_binary_weighted_score({0: True, 1: True}, rubric)

        self.assertEqual(score, 0.6)

    def test_healthbench_weighting_clamps_negative_score_to_zero(self):
        rubric = [
            {"description": "Includes advice.", "weight": 5},
            {"description": "Gives unsafe advice.", "weight": -10},
        ]

        score = judge._compute_binary_weighted_score({0: False, 1: True}, rubric)

        self.assertEqual(score, 0.0)

    def test_active_evolved_criterion_contributes_to_next_reward(self):
        evolver = RubricEvolver(OPSDConfig())
        parent = [{"title": "Parent", "description": "Parent criterion.", "weight": 10.0}]
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
                criterion_id="child_1",
                parent_index=0,
                parent_criterion_id="parent_0",
                title="Child",
                description="New discriminative criterion.",
                weight=2.0,
                source="decompose",
                status="active",
                proposed_step=1,
                verify_history=[{"passed": True}],
            )
        )

        next_batch = DataProto.from_dict(
            tensors={},
            non_tensors={
                "prompt_text": np.array(["Question"], dtype=object),
                "rubric": np.array([parent], dtype=object),
                "rubric_list": np.array([None], dtype=object),
            },
        )
        evolver.apply_active_rubric_to_batch(next_batch)
        reward_rubric = next_batch.non_tensor_batch["rubric"][0]

        self.assertEqual([item["criterion_id"] for item in reward_rubric], ["parent_0", "child_1"])
        self.assertAlmostEqual(
            judge._compute_binary_weighted_score({0: False, 1: True}, reward_rubric),
            2.0 / 12.0,
        )

    def test_normalizes_collated_numpy_rubric_and_prompt(self):
        rubric = np.array(
            [
                {"title": "Useful", "description": "Includes advice.", "weight": 10},
                {"title": "Unsafe", "description": "Gives unsafe advice.", "weight": -4},
            ],
            dtype=object,
        )
        prompt = np.array([{"role": "user", "content": "What should I do?"}], dtype=object)

        normalized = judge._normalize_rubric({"rubric": rubric})
        messages = judge._build_judge_messages(
            {"problem": prompt, "response": "Seek care."},
            normalized,
        )

        self.assertEqual(len(normalized), 2)
        self.assertIn("user: What should I do?", messages[1]["content"])
        self.assertIn("[Useful] Includes advice.", messages[1]["content"])

    def test_empty_rubric_raises_when_fail_on_error_is_enabled(self):
        with self.assertRaisesRegex(ValueError, "no rubric items"):
            judge._judge_one(
                {"problem": "question", "response": "answer"},
                model="judge",
                base_url="https://judge.example/v1",
                api_key_env="TEST_KEY",
                timeout=1,
                max_retries=0,
                response_retries=0,
                reasoning_effort=None,
                enable_thinking=False,
                temperature=0,
                fail_on_error=True,
            )

    def test_ark_provider_matches_volcengine_openai_compatible_endpoint(self):
        settings = judge.resolve_judge_settings(
            judge_provider="ark",
            model="deepseek-v4-flash-260425",
        )

        self.assertEqual(settings["base_url"], "https://ark.cn-beijing.volces.com/api/v3")
        self.assertEqual(settings["api_key_env"], "ARK_API_KEY")
        self.assertEqual(settings["model"], "deepseek-v4-flash-260425")
        self.assertIsNone(settings["reasoning_effort"])
        self.assertFalse(settings["enable_thinking"])

    def test_ark_request_uses_standard_chat_completion_shape(self):
        request = judge._build_judge_request(
            {
                "problem": "question",
                "response": "answer",
                "rubric": [{"description": "Answers the question.", "weight": 1}],
            },
            [{"description": "Answers the question.", "weight": 1}],
            model="deepseek-v4-flash-260425",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            reasoning_effort=None,
            enable_thinking=False,
            temperature=0.0,
            max_tokens=64,
            chat_template_kwargs=None,
            extra_body=None,
            response_format=None,
        )

        self.assertEqual(request["model"], "deepseek-v4-flash-260425")
        self.assertFalse(request["stream"])
        self.assertEqual(request["max_tokens"], 64)
        self.assertNotIn("extra_body", request)
        self.assertNotIn("response_format", request)
        self.assertNotIn("reasoning_effort", request)

    def test_deepseek_official_provider_defaults_to_non_thinking(self):
        settings = judge.resolve_judge_settings(
            judge_provider="deepseek",
            model="deepseek-v4-flash",
        )

        self.assertEqual(settings["base_url"], "https://api.deepseek.com")
        self.assertEqual(settings["api_key_env"], "DEEPSEEK_API_KEY")
        self.assertEqual(settings["model"], "deepseek-v4-flash")
        self.assertIsNone(settings["reasoning_effort"])
        self.assertFalse(settings["enable_thinking"])

    def test_deepseek_official_request_disables_thinking_explicitly(self):
        request = judge._build_judge_request(
            {
                "problem": "question",
                "response": "answer",
                "rubric": [{"description": "Answers the question.", "weight": 1}],
            },
            [{"description": "Answers the question.", "weight": 1}],
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            reasoning_effort=None,
            enable_thinking=False,
            temperature=0.0,
            max_tokens=64,
            chat_template_kwargs=None,
            extra_body=None,
            response_format={"type": "json_object"},
        )

        self.assertEqual(request["model"], "deepseek-v4-flash")
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning_effort", request)

    def test_failed_requests_are_not_cached_as_zero_rewards(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = RuntimeError("temporary failure")
        kwargs = {
            "model": "judge",
            "base_url": "https://judge.example/v1",
            "api_key_env": "TEST_KEY",
            "timeout": 1,
            "max_retries": 0,
            "response_retries": 0,
            "reasoning_effort": None,
            "enable_thinking": False,
            "temperature": 0,
            "fail_on_error": False,
        }
        inp = {
            "problem": "question",
            "response": "answer",
            "rubric": [{"description": "Answers the question.", "weight": 1}],
        }

        with mock.patch.object(judge, "_get_client", return_value=client):
            first = judge._judge_one(inp, **kwargs)
            second = judge._judge_one(inp, **kwargs)

        self.assertEqual(first["judge_error"], 1.0)
        self.assertEqual(second["judge_error"], 1.0)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertEqual(judge._CACHE, {})

    def test_extracts_first_complete_object_when_backend_appends_another_json(self):
        parsed = judge._extract_json_object(
            '{"1": "PRESENT", "2": "NOT_PRESENT"}\n{"metadata": "extra"}'
        )

        self.assertEqual(parsed, {"1": "PRESENT", "2": "NOT_PRESENT"})

    def test_extracts_presence_pairs_from_json_like_response_missing_commas(self):
        parsed = judge._extract_json_object(
            '{\n'
            '  "1": "PRESENT"\n'
            '  "2": "NOT_PRESENT"\n'
            '  "3": "PRESENT"\n'
            '}'
        )

        self.assertEqual(parsed, {"1": "PRESENT", "2": "NOT_PRESENT", "3": "PRESENT"})

    def test_local_vllm_request_disables_qwen_thinking_and_caps_tokens(self):
        valid = mock.Mock()
        valid.choices = [mock.Mock(message=mock.Mock(content='{"1": "PRESENT"}'))]
        client = mock.Mock()
        client.chat.completions.create.return_value = valid

        with mock.patch.object(judge, "_get_client", return_value=client):
            result = judge._judge_one(
                {
                    "problem": "question",
                    "response": "answer",
                    "rubric": [{"description": "Answers the question.", "weight": 1}],
                },
                model="judge",
                base_url="http://127.0.0.1:8008/v1",
                api_key_env="dummy-key",
                timeout=1,
                max_retries=0,
                response_retries=0,
                reasoning_effort=None,
                enable_thinking=False,
                temperature=0,
                fail_on_error=True,
                max_tokens=64,
            )

        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(result["judge_error"], 0.0)
        self.assertEqual(request["max_tokens"], 64)
        self.assertEqual(request["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(request["response_format"], {"type": "json_object"})

    def test_remote_non_thinking_provider_does_not_get_vllm_template_kwargs(self):
        extra_body = judge._build_extra_body(
            base_url="https://api.poixe.com/v1",
            enable_thinking=False,
            chat_template_kwargs=None,
            extra_body=None,
        )

        self.assertIsNone(extra_body)

    def test_retries_malformed_response_before_failing_training(self):
        malformed = mock.Mock()
        malformed.choices = [mock.Mock(message=mock.Mock(content="not json"))]
        valid = mock.Mock()
        valid.choices = [mock.Mock(message=mock.Mock(content='{"1": "PRESENT"}'))]
        client = mock.Mock()
        client.chat.completions.create.side_effect = [malformed, valid]

        with mock.patch.object(judge, "_get_client", return_value=client):
            result = judge._judge_one(
                {
                    "problem": "question",
                    "response": "answer",
                    "rubric": [{"description": "Answers the question.", "weight": 1}],
                },
                model="judge",
                base_url="https://judge.example/v1",
                api_key_env="TEST_KEY",
                timeout=1,
                max_retries=0,
                response_retries=1,
                reasoning_effort=None,
                enable_thinking=False,
                temperature=0,
                fail_on_error=True,
            )

        self.assertEqual(result["overall"], 1.0)
        self.assertEqual(result["judge_error"], 0.0)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_endpoint_pool_distributes_parallel_sample_requests(self):
        valid = mock.Mock()
        valid.choices = [mock.Mock(message=mock.Mock(content='{"1": "PRESENT"}'))]
        client_a = mock.Mock()
        client_a.chat.completions.create.return_value = valid
        client_b = mock.Mock()
        client_b.chat.completions.create.return_value = valid
        seen_urls = []

        def fake_get_client(api_key_env, base_url, timeout, max_retries):
            seen_urls.append(base_url)
            return {
                "https://judge-a.example/v1": client_a,
                "https://judge-b.example/v1": client_b,
            }[base_url]

        reward_inputs = [
            {
                "problem": "question",
                "response": "answer a",
                "rubric": [{"description": "Answers the question.", "weight": 1}],
            },
            {
                "problem": "question",
                "response": "answer b",
                "rubric": [{"description": "Answers the question.", "weight": 1}],
            },
        ]

        with mock.patch.object(judge, "_get_client", side_effect=fake_get_client):
            results = judge.compute_score(
                reward_inputs,
                judge_provider=None,
                model="judge",
                base_urls=["https://judge-a.example/v1", "https://judge-b.example/v1"],
                api_key_env="TEST_KEY",
                timeout=1,
                max_retries=0,
                response_retries=0,
                enable_thinking=False,
                max_workers=8,
                max_workers_per_url=1,
                fail_on_error=True,
            )

        self.assertEqual([result["overall"] for result in results], [1.0, 1.0])
        self.assertEqual(set(seen_urls), {"https://judge-a.example/v1", "https://judge-b.example/v1"})

    def test_can_disable_criterion_metrics(self):
        valid = mock.Mock()
        valid.choices = [mock.Mock(message=mock.Mock(content='{"1": "PRESENT"}'))]
        client = mock.Mock()
        client.chat.completions.create.return_value = valid

        with mock.patch.object(judge, "_get_client", return_value=client):
            results = judge.compute_score(
                [
                    {
                        "problem": "question",
                        "response": "answer",
                        "rubric": [{"description": "Answers the question.", "weight": 1}],
                    }
                ],
                judge_provider=None,
                model="judge",
                base_url="https://judge.example/v1",
                api_key_env="TEST_KEY",
                timeout=1,
                max_retries=0,
                response_retries=0,
                enable_thinking=False,
                emit_criterion_scores=False,
                fail_on_error=True,
            )

        self.assertEqual(results[0]["overall"], 1.0)
        self.assertNotIn("criterion_0_score", results[0])
        self.assertNotIn("criterion_count", results[0])

    def test_effective_workers_can_be_controlled_per_url(self):
        self.assertEqual(
            judge._effective_worker_count(
                task_count=100,
                url_count=3,
                max_workers=128,
                max_workers_per_url=8,
            ),
            24,
        )
        self.assertEqual(
            judge._effective_worker_count(
                task_count=100,
                url_count=3,
                max_workers=16,
                max_workers_per_url=8,
            ),
            16,
        )


if __name__ == "__main__":
    unittest.main()
