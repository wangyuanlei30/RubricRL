import unittest

from tensordict import TensorDict

from verl.protocol import DataProto
from verl.utils import torch_functional as VF
from verl.workers.rollout.vllm_rollout_spmd import _resolve_response_padding_length


class RolloutResponsePaddingTest(unittest.TestCase):
    def test_request_override_expands_dp_stable_response_width(self):
        self.assertEqual(_resolve_response_padding_length(4096, None), 4096)
        self.assertEqual(_resolve_response_padding_length(4096, 1024), 4096)
        self.assertEqual(_resolve_response_padding_length(4096, 8192), 8192)

    def test_worker_outputs_with_different_local_maxima_can_concat(self):
        target_length = _resolve_response_padding_length(4096, 8192)
        worker_a_responses = VF.pad_2d_list_to_length(
            [list(range(4519))],
            pad_token_id=0,
            max_length=target_length,
        )
        worker_b_responses = VF.pad_2d_list_to_length(
            [list(range(8048))],
            pad_token_id=0,
            max_length=target_length,
        )
        worker_outputs = [
            DataProto(
                batch=TensorDict(
                    {"responses": responses},
                    batch_size=responses.size(0),
                )
            )
            for responses in (worker_a_responses, worker_b_responses)
        ]

        combined = DataProto.concat(worker_outputs)

        self.assertEqual(tuple(combined.batch["responses"].shape), (2, 8192))


if __name__ == "__main__":
    unittest.main()
