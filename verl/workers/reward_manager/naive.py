# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


_PARALLEL_THRESHOLD = int(os.environ.get("VERL_REWARD_PARALLEL_THRESHOLD", "256"))
_PARALLEL_MAX_WORKERS = int(
    os.environ.get("VERL_REWARD_NUM_WORKERS", str(min(32, max(1, (os.cpu_count() or 4) // 2))))
)

_POOL = None
_POOL_KEY = None
_WORKER_FN = None


def _worker_init(fn_bytes: bytes) -> None:
    import cloudpickle

    global _WORKER_FN
    _WORKER_FN = cloudpickle.loads(fn_bytes)


def _worker_score(args: tuple) -> Any:
    data_source, response_str, ground_truth, extra_info = args
    return _WORKER_FN(
        data_source=data_source,
        solution_str=response_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )


def _get_or_create_pool(compute_score) -> ProcessPoolExecutor:
    global _POOL, _POOL_KEY
    key = id(compute_score)
    if _POOL is not None and _POOL_KEY == key:
        return _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = None
    import cloudpickle

    fn_bytes = cloudpickle.dumps(compute_score)
    ctx = mp.get_context("spawn")
    _POOL = ProcessPoolExecutor(
        max_workers=_PARALLEL_MAX_WORKERS,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(fn_bytes,),
    )
    _POOL_KEY = key
    return _POOL


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        n = len(data)
        valid_response_lengths: list[int] = [0] * n
        score_args: list[tuple] = [None] * n  # type: ignore[list-item]
        examine_payloads: list[tuple] = []  # (data_source, prompt_ids, response_str)
        examine_budget: dict[Any, int] = defaultdict(int)

        for i in range(n):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]
            valid_response_lengths[i] = valid_response_length

            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            score_args[i] = (data_source, response_str, ground_truth, extra_info)

            if examine_budget[data_source] < self.num_examine:
                examine_budget[data_source] += 1
                valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                examine_payloads.append((data_source, valid_prompt_ids, response_str, ground_truth))

        scores = self._compute_scores(score_args)

        for i, score in enumerate(scores):
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            valid_response_length = valid_response_lengths[i]
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = reward

        for data_source, prompt_ids_t, response_str, ground_truth in examine_payloads:
            prompt_str = self.tokenizer.decode(prompt_ids_t, skip_special_tokens=True)
            print("[prompt]", prompt_str)
            print("[response]", response_str)
            print("[ground_truth]", ground_truth)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def _compute_scores(self, score_args: list[tuple]) -> list[Any]:
        n = len(score_args)
        if n < _PARALLEL_THRESHOLD or _PARALLEL_MAX_WORKERS <= 1:
            return [
                self.compute_score(
                    data_source=ds,
                    solution_str=resp,
                    ground_truth=gt,
                    extra_info=ei,
                )
                for (ds, resp, gt, ei) in score_args
            ]
        try:
            pool = _get_or_create_pool(self.compute_score)
            return list(pool.map(_worker_score, score_args, chunksize=max(1, n // (_PARALLEL_MAX_WORKERS * 4))))
        except Exception as e:
            print(f"[NaiveRewardManager] parallel scoring failed ({e!r}), falling back to serial")
            return [
                self.compute_score(
                    data_source=ds,
                    solution_str=resp,
                    ground_truth=gt,
                    extra_info=ei,
                )
                for (ds, resp, gt, ei) in score_args
            ]
