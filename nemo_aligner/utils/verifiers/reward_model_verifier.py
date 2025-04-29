from typing import List, Dict, Any
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# Configure logging with a more specific setup
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class RewardModelVerifier:
    def __init__(self, model_name: str = "nvidia/Llama-3.1-Nemotron-70B-Reward-HF", 
                 device: str = "auto", 
                 dtype: torch.dtype = torch.bfloat16):
        """
        Initialize Reward Model verifier
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run the model on ("cuda", "cpu")
            dtype: Data type for model weights
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=dtype, 
                device_map=device
            )

        except Exception as e:
            logger.error(f"Error initializing reward model: {str(e)}")
            raise

    def extract_dialogue_llama(self, text):
        user_pattern = r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|start_header_id\|>"
        assistant_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|start_header_id\|>"

        user_text = re.findall(user_pattern, text, re.DOTALL)
        assistant_text = re.findall(assistant_pattern, text, re.DOTALL)
        
        return user_text, assistant_text

    def verify(self, prompts: List[str], responses: List[str]) -> List[Dict[str, Any]]:
        """
        Verify multiple responses using the reward model
        
        Args:
            prompts: List of problem prompts
            responses: List of model responses to verify
            
        Returns:
            List of dictionaries containing verification results with reward scores
        """
        results = []
        
        for prompt, response in zip(prompts, responses):
            try:
                
                # Extract all previous turns from the prompt
                user_text, assistant_text = self.extract_dialogue_llama(prompt)
                assert len(user_text) == len(assistant_text) +1, f"user_text: {user_text}, assistant_text: {assistant_text}"
                # Format as chat messages
                messages = []
                for i in range(len(assistant_text)):
                    messages.append({'role': "user", "content": user_text[i]})
                    messages.append({'role': "assistant", "content": assistant_text[i]})
                    
                messages.append({'role': "user", "content": user_text[-1]})
                messages.append({'role': "assistant", "content": response})
                print(messages)
                logger.info(f"Messages: {messages}")  # Use the module logger instead
                # Tokenize the conversation
                tokenized_message = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    add_generation_prompt=False, 
                    return_tensors="pt", 
                    return_dict=True
                )
                # print(tokenized_message)
                # logger.info(f"Tokenized message: {tokenized_message}")  # Use the module logger
                # Move tensors to the correct device
                input_ids = tokenized_message['input_ids'].cuda()
                attention_mask = tokenized_message['attention_mask'].cuda()
                # logger.info(f"Input IDs: {input_ids}")  # Use the module logger
                # logger.info(f"Attention mask: {attention_mask}")  # Use the module logger
                # Generate with the model to get reward score
                with torch.no_grad():
                    response_token_ids = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                # logger.info(f"Response token IDs: {response_token_ids}")  # Use the module logger
                # Extract the reward score
                reward_score = response_token_ids['scores'][0][0][0].item()
                
                print(reward_score)
                logger.info(f"Reward score: {reward_score}")  # Use the module logger
                # Normalize the reward score to [0, 1] range if needed
                # This is a placeholder - you may need to adjust based on your model's output range
                # normalized_score = (reward_score + 5) / 10  # Example normalization
                
                results.append({
                    "reward_score": reward_score
                })
                
            except Exception as e:
                logger.error(f"Error processing prompt-response pair: {str(e)}, prompt: {prompt}, response: {response}")
                # Return a default low score in case of error
                results.append({
                    "reward_score": -10.0,  # Very low score for errors
                })
        
        return results