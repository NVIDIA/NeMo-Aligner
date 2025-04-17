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

class GenRMFormatChecker:
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
    def check_genrm_format(prompt: str, response: str, num_responses: int) -> bool:
        """Check if response format matches the genrm requirement in prompt."""
        
        response = response.split("</think>")[-1].strip()
        
        # Track positions to verify correct ordering
        last_position = -1
        
        # Check for analysis sections in correct order
        for i in range(1, num_responses + 1):
            begin_marker = f"[The Begin of Analysis on Response {i}]"
            end_marker = f"[The End of Analysis on Response {i}]"
            
            if begin_marker not in response or end_marker not in response:
                return False
            
            # Check proper ordering
            begin_pos = response.find(begin_marker)
            end_pos = response.find(end_marker)
            
            if begin_pos <= last_position or begin_pos >= end_pos:
                return False
                
            last_position = end_pos
        
        # Check for individual scores section
        begin_scores = "[The Begin of Individual Scores]"
        end_scores = "[The End of Individual Scores]"
        
        if begin_scores not in response or end_scores not in response:
            return False
        
        # Check proper ordering for scores section
        begin_scores_pos = response.find(begin_scores)
        end_scores_pos = response.find(end_scores)
        
        if begin_scores_pos <= last_position or begin_scores_pos >= end_scores_pos:
            return False
            
        last_position = end_scores_pos
        
        # Check for ranking score if there are 2 responses
        if num_responses == 2:
            begin_ranking = "[The Begin of Ranking Score]"
            end_ranking = "[The End of Ranking Score]"
            
            if begin_ranking not in response or end_ranking not in response:
                return False
            
            # Check proper ordering for ranking section
            begin_ranking_pos = response.find(begin_ranking)
            end_ranking_pos = response.find(end_ranking)
            
            if begin_ranking_pos <= last_position or begin_ranking_pos >= end_ranking_pos:
                return False
        
        return True

    @staticmethod
    def calculate_format_metrics(prompts: List[str], responses: List[str], is_end: List[bool], num_responses_list: List[int]):
        """Calculate format rewards and metrics for a batch of responses."""
        format_rewards = []
        
        
        for prompt, response, is_end, num_responses in zip(prompts, responses, is_end, num_responses_list):
            # Check format compliance
            # only check for those correctly terminated responses
            if is_end:
                thinking_format_correct = GenRMFormatChecker.check_thinking_format(prompt, response)
                genrm_format_correct = GenRMFormatChecker.check_genrm_format(prompt, response, num_responses)
                format_correct = thinking_format_correct and genrm_format_correct
                format_rewards.append(1.0 if format_correct else 0.0)
            else:
                format_rewards.append(1.0)
        
        return format_rewards