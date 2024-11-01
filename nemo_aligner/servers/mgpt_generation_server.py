# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
"""Utilities for generating text."""

import json
import threading

import torch
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.modules.common.text_generation_server import MegatronGenerate, GENERATE_NUM, API_ALLOWED_KEYS, lock
from nemo.utils import logging




# class TextContent(BaseModel):
#     type: str
#     text: str

#     @field_validator('type')
#     def check_type(cls, v):
#         if v != 'text':
#             raise ValueError('type must be text')
#         return v

# class ImageUrl(BaseModel):
#     url: str

# class ImageContent(BaseModel):
#     type: str
#     image_url: ImageUrl

#     @field_validator('type')
#     def check_type(cls, v):
#         if v != 'image_url':
#             raise ValueError('type must be image_url')
#         return v

# class Message(BaseModel):
#     role: str
#     content: Union[str, List[Union[TextContent, ImageContent]]]

# class ChatCompletionRequest(BaseModel):
#     model: Optional[str] = "mock-gpt-model"
#     messages: List[Message]
#     max_tokens: Optional[int] = 1024
#     temperature: Optional[float] = 0.2
#     stream: Optional[bool] = False



class MegatronMultimodalGenerate(MegatronGenerate):
    def __init__(self, model, inference_strategy=None):
        super().__init__(model, inference_strategy)
        self.image_processor = self.model.model.module.image_processor if hasattr(self.model.model, "module") else self.model.model.image_processor

    def put(self):
        logging.info("request IP: " + str(request.remote_addr))
        #logging.info(json.dumps(request.get_json()))
        # check keys
        for key in request.get_json().keys():
            if key not in API_ALLOWED_KEYS:
                logging.error(f"The request key {key} is not allowed")

        sentences = request.get_json()["sentences"]
        if isinstance(sentences, tuple):  # Input can be text or tensor
            if len(sentences[0]) != len(sentences[1]) or sentences[0] > 128:
                return "Maximum number of sentences is 128", 400
        elif len(sentences) > 128:
            return "Maximum number of sentences is 128", 400

        task_ids = None  # Used for ptuned/prompt tuned models only
        if "task_ids" in request.get_json():
            task_ids = request.get_json()["task_ids"]
            if not isinstance(sentences, tuple):
                return "Input at 'sentences' must by a tuple of two tensors like:\
                    (context_tokens_tensor, context_length_tensor) if task ids are given"
            if len(task_ids) != len(sentences[0]):
                return "Each sentence must have a corresponding task id for p-tuned/prompt-tuned models"

        tokens_to_generate = 64  # Choosing hopefully sane default.  Full sequence is slow
        if "tokens_to_generate" in request.get_json():
            tokens_to_generate = request.get_json()["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int):
                return "tokens_to_generate must be an integer greater than 0"
            if tokens_to_generate < 1:
                return "tokens_to_generate must be an integer greater than 0"

        all_probs = False
        if "all_probs" in request.get_json():
            all_probs = request.get_json()["all_probs"]
            if not isinstance(all_probs, bool):
                return "all_probs must be a boolean value"

        temperature = 1.0
        if "temperature" in request.get_json():
            temperature = request.get_json()["temperature"]
            if not (type(temperature) == int or type(temperature) == float):
                return "temperature must be a positive number less than or equal to 100.0"
            if not (0.0 < temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"

        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"

        greedy = False
        if "greedy" in request.get_json():
            greedy = request.get_json()["greedy"]
            if not isinstance(greedy, bool):
                return "greedy must be a boolean value"

        top_k = 0
        if "top_k" in request.get_json():
            top_k = request.get_json()["top_k"]
            if not (type(top_k) == int or type(top_k) == float):
                return "top_k must be a positive integer number"
            if not (0 <= top_k):
                return "top_k must be a positive integer number"

        top_p = 0.9
        if "top_p" in request.get_json():
            top_p = request.get_json()["top_p"]
            if not (type(top_p) == int or type(top_p) == float):
                return "top_p must be a positive number less than or equal to 1.0"
            if not (0.0 <= top_p <= 1.0):
                return "top_p must be a positive number less than or equal to 1.0"

        repetition_penalty = 1.2
        if "repetition_penalty" in request.get_json():
            repetition_penalty = request.get_json()["repetition_penalty"]
            if not (type(repetition_penalty) == int or type(repetition_penalty) == float):
                return "repetition_penalty must be a positive number no less than 1.0"
            if not (1.0 <= repetition_penalty):
                return "repetition_penalty must be a positive number no less than 1.0"

        end_strings = ['<|endoftext|>']
        if 'end_strings' in request.get_json():
            end_strings = request.get_json()['end_strings']
            if not isinstance(end_strings, list):
                return "expect end_strings to be a list of strings"
            if not all([isinstance(s, str) for s in end_strings]):
                return "expect end_strings to be a list of strings"

        min_tokens_to_generate = 0
        if "min_tokens_to_generate" in request.get_json():
            min_tokens_to_generate = request.get_json()["min_tokens_to_generate"]
            if not isinstance(min_tokens_to_generate, int):
                return "min_tokens_to_generate must be an integer no less than 0"
            if min_tokens_to_generate < 0:
                return "min_tokens_to_generate must be an integer no less than 0"

        neighbors = None
        if "neighbors" in request.get_json():
            neighbors = request.get_json()["neighbors"]
            if not isinstance(neighbors, int):
                return "num of neighbors must be an integer no less than 0"
            if neighbors < 0:
                return "num of neighbors must be an integer no less than 0"

        compute_logprob = False
        if "compute_logprob" in request.get_json():
            compute_logprob = request.get_json()["compute_logprob"]
            if not isinstance(compute_logprob, bool):
                return "compute_logprob must be a boolean value"

        random_seed = None
        if "random_seed" in request.get_json():
            random_seed = request.get_json()["random_seed"]
            if random_seed is not None and not isinstance(random_seed, int):
                return "random_seed must be a positive integer number or None"
            if random_seed is not None and random_seed < 0:
                return "random_seed must be a positive integer number or None"

        image_list = None
        if "images" in request.get_json():
            base64_images = request.get_json()["images"]
            image_list = []
            for base64_string in base64_images:
                try:
                    # Decode the base64 string
                    img_data = base64.b64decode(base64_string)
                    # Open the image using PIL
                    img = Image.open(BytesIO(img_data))
                    img.load()
                    img = img.convert("RGB")
                    # Append to image_list
                    image_list.append(img)
                except Exception as e:
                    logging.error(f"Error processing image: {str(e)}")
                    return f"Error processing image: {str(e)}", 400


        with lock:  # Need to get lock to keep multiple threads from hitting code
            MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
            extra = {}
            if task_ids is not None:
                extra['task_ids'] = task_ids
            if self.inference_strategy is not None:
                extra['strategy'] = self.inference_strategy
            
            image_sizes = None
            if image_list is not None:
                image_inputs = self.image_processor(image_list, return_tensors="pt")
                image_list, image_sizes = image_inputs["pixel_values"], image_inputs["image_sizes"]

            context_tokens_tensor, context_length_tensor = self.inference_strategy.tokenize_batch(
                sentences, tokens_to_generate, add_BOS, image_sizes=image_sizes
            )

            output = generate(
                model=self.model,
                inputs=(context_tokens_tensor, context_length_tensor),
                tokens_to_generate=tokens_to_generate,
                all_probs=all_probs,
                temperature=temperature,
                add_BOS=add_BOS,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
                compute_attention_mask=True,
                compute_logprob=False,
                repetition_penalty=repetition_penalty,
                end_strings=end_strings,
                image_list=image_list,
                min_tokens_to_generate=min_tokens_to_generate,
                random_seed=random_seed,
                **extra,
            )
            for k in output:
                if isinstance(output[k], torch.Tensor):
                    output[k] = output[k].tolist()

            if output is not None:  # may be `None` for intermediate PP ranks when PP>2
                # adding predictions key which contains only model predictions without the prompt
                output["predictions"] = [
                    self.model.tokenizer.ids_to_text(tokens[length.item() :][:tokens_to_generate])
                    for tokens, length in zip(output["token_ids"], context_length_tensor)
                ]
                
        if not all_probs:
            del output['full_logprob']

        return jsonify(output)


async def process_queue():
    while True:
        batch = []
        while len(batch) < 4:
            try:
                request_entry = await asyncio.wait_for(request_queue.get(), timeout=0.05)
                batch.append(request_entry)
            except asyncio.TimeoutError:
                break
        
        if batch:
            # Text-only for now
            def get_prompt(messages):
                # prompt template gets applied in NeMo already
                # so we only add user content and filter out system
                prompt, first = "", True
                for message in messages:
                    if message.role.lower() == "system": continue

                    if not first: prompt += f"<extra_id_1>{message.role.capitalize()}\n"

                    if isinstance(message.content, str):
                        prompt += message.content.strip() + "\n"
                    else:
                        for content in message.content:
                            if content.type == "text":
                                prompt += content.text.strip() + "\n"
                            elif content.type == "image_url":
                                prompt += "<image>\n"
                            else:
                                raise ValueError(f"Unsupported content type: {content.type}")
                    
                    first = False

                return prompt.strip()

            def get_image(messages):
                image = None
                for message in messages:
                    if not isinstance(message.content, str):
                        for content in message.content:
                            if content.type == "image_url":
                                if image is not None:
                                    raise ValueError("Multiple images in the message")

                                url = content.image_url.url
                                logging.info(f"Got image url {url} downloading...")
                                image = PIL.Image.open(requests.get(url, stream=True).raw)
                                logging.info(f"Got image size {image.size}")

                if image is not None:
                    return image_processor(image)
                
                return None

            input_prompts = [{
                                "prompt": get_prompt(entry["request"].messages),
                                "label" : "helpfulness:5,correctness:5,coherence:5",
                                "image" : get_image(entry["request"].messages)
                            } for entry in batch]
            responses = predict_impl(input_prompts)
            
            for entry, response in zip(batch, responses):
                response_content = response["clean_response"].replace("<extra_id_1>", "").strip()
                entry["response"].set_result({
                    "id": str(time.time()),
                    "object": "chat.completion",
                    "created": time.time(),
                    "model": entry["request"].model,
                    "choices": [{"message": Message(role="assistant", content=response_content)}],
                })
                logging.info(f"Request {entry['id']} response prepared")



class MegatronMultimodalServer(object):
    def __init__(self, model, inference_strategy=None):
        self.app = Flask(__name__, static_url_path='')
        api = Api(self.app)
        api.add_resource(MegatronMultimodalGenerate, '/generate', resource_class_args=[model, inference_strategy])
        
        #self.fast_app = FastAPI(title="OpenAI-compatible API")
        #self.fast_app.mount("/generate", WSGIMiddleware(flask_app))        


    def run(self, url, port=5000):
        #request_queue = asyncio.Queue()
        self.app.run(url, threaded=True, port=port, debug=False)