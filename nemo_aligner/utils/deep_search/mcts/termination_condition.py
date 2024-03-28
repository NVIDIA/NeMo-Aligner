# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import re


class TerminationCondition:
    def __init__(self, max_depth, end_strings, end_tokens):
        self.end_strings = end_strings
        self.max_depth = max_depth
        self.end_tokens = set(end_tokens)

    def ends_by_end_strings(self, text, tokens):
        if tokens[-1] in self.end_tokens:
            return True
        for end_string in self.end_strings:
            if text.endswith(end_string):
                return True
        return False

    def ends_by_depth(self, depth):
        return depth >= self.max_depth

    def __call__(self, text, depth, tokens):
        if self.ends_by_depth(depth):
            return True
        return self.ends_by_end_strings(text, tokens)

    def has_answer(self, response):
        response = response.lower()
        # predicted answer matches the answer pattern
        numbers = re.findall(r"\{\{([\d,]+)\}\}", response)
        # Extract the last number
        last_number = numbers[-1] if numbers else None
        if last_number is not None:
            return True
        return False
