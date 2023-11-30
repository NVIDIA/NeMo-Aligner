# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
from nemo_aligner.utils.trainer_utils import check_progress, compute_limit_batches


@pytest.mark.parametrize(
    "number_of_batches,limit_batches,expected_result",
    [(1000, 0, 0), (1000, 1.0, 1000), (1000, 0.5, 500), (1000, 10, 10)],
)
def test_compute_limit_batches(number_of_batches, limit_batches, expected_result):
    result = compute_limit_batches(number_of_batches, limit_batches)
    assert (
        expected_result == result
    ), f"{expected_result=} is different than {result=} with input {number_of_batches=} {limit_batches=}"


CHECK_PROGRESS_ARGS = [
    (
        {"step": 10, "max_steps": 10, "val_check_interval": 3, "save_interval": 9, "limit_val_batches": 1.0},
        (True, True, True),
    ),
    (
        {"step": 10, "max_steps": 10, "val_check_interval": 3, "save_interval": 9, "limit_val_batches": 0},
        (False, True, True),
    ),
    (
        {"step": 10, "max_steps": 10, "val_check_interval": 3, "save_interval": 9, "limit_val_batches": 10},
        (True, True, True),
    ),
    (
        {"step": 3, "max_steps": 10, "val_check_interval": 3, "save_interval": 9, "limit_val_batches": 10},
        (True, False, False),
    ),
    (
        {"step": 6, "max_steps": 10, "val_check_interval": 3, "save_interval": 9, "limit_val_batches": 10},
        (True, False, False),
    ),
    (
        {"step": 9, "max_steps": 10, "val_check_interval": 3, "save_interval": 9, "limit_val_batches": 10},
        (True, True, False),
    ),
    (
        {"step": 3, "max_steps": 10, "val_check_interval": 3, "save_interval": 9, "limit_val_batches": 0},
        (False, False, False),
    ),
    (
        {"step": 6, "max_steps": 10, "val_check_interval": 3, "save_interval": 9, "limit_val_batches": 0},
        (False, False, False),
    ),
    (
        {"step": 9, "max_steps": 10, "val_check_interval": 3, "save_interval": 9, "limit_val_batches": 0},
        (False, True, False),
    ),
    (
        {"step": 6, "max_steps": 10, "val_check_interval": 1, "save_interval": 2, "limit_val_batches": 0},
        (False, True, False),
    ),
    (
        {"step": 10, "max_steps": 10, "val_check_interval": 1, "save_interval": 2, "limit_val_batches": 0},
        (False, True, True),
    ),
    (
        {"step": 10, "max_steps": 10, "val_check_interval": 3, "save_interval": 0, "limit_val_batches": 10},
        (True, False, True),
    ),
    (
        {"step": 10, "max_steps": 10, "val_check_interval": 0, "save_interval": 9, "limit_val_batches": 10},
        (False, True, True),
    ),
    (
        {"step": 10, "max_steps": 10, "val_check_interval": 0, "save_interval": 0, "limit_val_batches": 10},
        (False, False, True),
    ),
]


@pytest.mark.parametrize("input_kwargs,expected_result", CHECK_PROGRESS_ARGS)
def test_check_progress(input_kwargs, expected_result):
    result = check_progress(**input_kwargs)
    assert expected_result == result, f"{expected_result=} is different than {result=} with input {input_kwargs=}"
