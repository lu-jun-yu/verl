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
"""
Reward function for ...</think><answer>...</answer> format.

Note: The prompt already contains "<think>\n" as generation_prefix,
so the response should be: "...thinking...</think><answer>answer</answer>"
"""

import re
from typing import Optional, Tuple


def check_format(response_str: str) -> bool:
    """Check if the response follows the strict format: ...思考内容...</think><answer>答案</answer>

    The response should:
    1. End the thinking section with </think>
    2. Have <answer>...</answer> after </think>

    Args:
        response_str: The response string to check (without the <think> prefix).

    Returns:
        True if the format is correct, False otherwise.
    """
    # Strict format: must have </think> followed by <answer>...</answer>
    pattern = r'^.*</think>\s*<answer>.*</answer>\s*$'
    return bool(re.match(pattern, response_str, re.DOTALL))


def extract_answer(response_str: str) -> Optional[str]:
    """Extract the answer from <answer>...</answer> tags.

    Args:
        response_str: The response string.

    Returns:
        The answer content, or None if not found.
    """
    match = re.search(r'<answer>(.*?)</answer>', response_str, re.DOTALL)
    return match.group(1).strip() if match else None


def normalize_answer(answer: str) -> str:
    """Normalize the answer for comparison.

    Args:
        answer: The answer string to normalize.

    Returns:
        Normalized answer string.
    """
    answer = answer.strip()
    # Remove $, commas, and spaces
    answer = re.sub(r'[\$,\s]', '', answer)
    # Handle boxed answers: \boxed{...}
    boxed_match = re.search(r'\\boxed\{(.*?)\}', answer)
    if boxed_match:
        answer = boxed_match.group(1)
    return answer


def compute_score(
    solution_str: str,
    ground_truth: str,
    format_reward: float = 0.5,
    correct_reward: float = 0.5,
) -> float:
    """Compute the score for a response.

    Total reward = format_reward + correct_reward (if both conditions met)

    Args:
        solution_str: The response string (without <think> prefix, which is in prompt).
        ground_truth: The expected answer.
        format_reward: Reward for correct format (0.5).
        correct_reward: Reward for correct answer (0.5).

    Returns:
        The computed score (0, 0.5, or 1.0).
    """
    total_score = 0.0

    # Check format: must have </think><answer>...</answer>
    if check_format(solution_str):
        total_score += format_reward

    # Check answer correctness
    answer_content = extract_answer(solution_str)
    if answer_content is not None:
        normalized_answer = normalize_answer(answer_content)
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_answer == normalized_ground_truth:
            total_score += correct_reward

    return total_score
