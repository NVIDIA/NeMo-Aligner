# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Tuple

class FormatChecker:
    @staticmethod
    def check_thinking_format(prompt: str, response: str) -> bool:
        """Check if response format matches the thinking requirement in prompt."""
        thinking_on = "detailed thinking on" in prompt.lower()
        thinking_off = "detailed thinking off" in prompt.lower()
        
        # For thinking off, ensure no think tags exist
        if thinking_off:
            return "<think>" not in response and "</think>" not in response
        
        # For thinking on, check proper tag format
        if thinking_on:
            # Count occurrences
            start_count = response.count("<think>")
            end_count = response.count("</think>")
            
            # Check if tags appear exactly once
            if start_count != 1 or end_count != 1:
                return False
            
            # Check proper ordering
            start_pos = response.find("<think>")
            end_pos = response.find("</think>")
            
            return start_pos < end_pos
            
        return True  # No format requirements if neither flag is present
    
    @staticmethod
    def calculate_format_metrics(prompts: List[str], responses: List[str], is_end: List[bool]):
        """Calculate format rewards and metrics for a batch of responses."""
        format_rewards = []
        
        for prompt, response, is_end in zip(prompts, responses, is_end):
            # Check format compliance
            # only check for those correctly terminated responses
            if is_end:
                format_correct = FormatChecker.check_thinking_format(prompt, response)
                format_rewards.append(1.0 if format_correct else 0.0)
            else:
                format_rewards.append(1.0)
            
        
        return format_rewards