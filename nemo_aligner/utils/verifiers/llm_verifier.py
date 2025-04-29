from typing import List, Dict, Any
import logging
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

class LLMVerifier:
    def __init__(self, api_base: str = "http://localhost:5000/v1", max_tokens: int = 32, temperature: float = 0.0):
        """
        Initialize LLM verifier that connects to vLLM OpenAI-compatible API
        
        Args:
            api_base: Base URL of the vLLM server
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.client = OpenAI(base_url=api_base, api_key="dummy")  # vLLM doesn't check API key
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.judge_prompt_template = """Problem:
{prompt}

Ground Truth Answer:
{ground_truth}

Model Response:
{response}

Instructions:
You are a fair and consistent judge. Above is a problem, a ground truth answer, and a model's response. Evaluate if the model's response matches the ground truth answer.
Answer "YES" or "NO", then provide a brief explanation.
"""

    def _call_single_api(self, msg: List[Dict[str, str]]) -> str:
        """Make a single API call to the vLLM server"""
        try:
            completion = self.client.chat.completions.create(
                model="model",
                messages=msg,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=600
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in single API call: {str(e)}")
            raise

    def _call_vllm_api(self, prompts: List[str]) -> List[str]:
        """Make batch requests to the vLLM server using chat completion with multithreading"""
        try:
            messages = [
                [{"role": "user", "content": prompt}] for prompt in prompts
            ]
            
            # Use ThreadPoolExecutor for parallel API calls
            with ThreadPoolExecutor(max_workers=min(32, len(messages))) as executor:
                futures = [executor.submit(self._call_single_api, msg) for msg in messages]
                results = [future.result() for future in as_completed(futures)]
            
            # Sort results back to original order
            sorted_results = [None] * len(prompts)
            for idx, future in enumerate(futures):
                sorted_results[idx] = future.result()
                
            return sorted_results
            
        except Exception as e:
            logging.error(f"Error in batch API calls: {str(e)}")
            raise

    def verify(self, prompts: List[str], responses: List[str], 
               ground_truths: List[str]) -> List[Dict[str, Any]]:
        """
        Verify multiple responses using the LLM judge
        
        Args:
            prompts: List of problem prompts
            responses: List of model responses to verify
            ground_truths: List of ground truth answers
            
        Returns:
            List of dictionaries containing verification results
        """
        results = []
        # to debug with exact match
        # for resp, gt in zip(responses, ground_truths):
        #     if resp.lower() == gt.lower():
        #         results.append({
        #             "passed": True,
        #             "full_response": resp
        #         })
        #     else:
        #         results.append({
        #             "passed": False,
        #             "full_response": resp
        #         })

        # Prepare prompts for the judge
        judge_prompts = [
            self.judge_prompt_template.format(
                prompt=prompt,
                ground_truth=gt,
                response=resp
            )
            for prompt, gt, resp in zip(prompts, ground_truths, responses)
        ]
        
        # Get LLM judgments through API
        generated_texts = self._call_vllm_api(judge_prompts)
        print(generated_texts)
        # Process results
        for gt, resp, generated_text in zip(ground_truths, responses, generated_texts):
            generated_text = generated_text.strip()
            is_correct = "yes" in generated_text.lower() and "no" not in generated_text.lower()
            score = 1 if is_correct else 0
                    
            results.append({
                "passed": score,
                "full_response": generated_text
            })
            
        return results