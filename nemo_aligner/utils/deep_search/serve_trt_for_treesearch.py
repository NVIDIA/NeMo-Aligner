import copy
import json
import sys
import time
import uuid
from argparse import ArgumentParser
from typing import List

import tensorrt_llm.bindings.executor as trtllm
import torch
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from mpi4py import MPI
from nemo_skills.inference.server.model import TensorRTLLMModel, postprocess_output, preprocess_request
from nemo_skills.inference.server.serve_trt import (
    TensorRTLLM,
    TrtGetResult,
    TrtStartGeneration,
    get_output,
    get_output_single,
    parse_input,
    remove_stop_tokens,
)

from nemo.utils import logging


def generate(
    runner,
    batch_input_ids: List[torch.Tensor],
    input_lengths,
    *,
    sampling_config=None,
    lora_uids=None,
    streaming: bool = False,
    stopping_criteria=None,
    logits_processor=None,
    max_new_tokens: int = 1,
    end_id: int | None = None,
    pad_id: int | None = None,
    bad_words_list: list[list[int]] | None = None,
    tokenizer=None,
    stop_words_list=None,
    return_dict: bool = False,
    output_sequence_lengths: bool = False,
    output_log_probs: bool = False,
    output_cum_log_probs: bool = False,
    prompt_table=None,
    prompt_tasks=None,
    **kwargs,
):
    """
    Generates sequences of token ids.
    The generation-controlling parameters are set in the sampling_config; it will be set to a default one if not passed.
    You can override any sampling_config's attributes by passing corresponding parameters.

    Args:
        batch_input_ids (List[torch.Tensor]):
            A list of input id tensors. Each tensor is of shape (sequence_length, ).
        sampling_config (SamplingConfig):
            The sampling configuration to be used as base parametrization for the generation call.
            The passed **kwargs matching the sampling_config's attributes will override them.
            If the sampling_config is not provided, a default will be used.
        prompt_table (str or torch.Tensor):
            The file path of prompt table (.npy format, exported by nemo_prompt_convert.py) or the prompt table itself.
        prompt_tasks (str):
            The prompt tuning task ids for the input batch, in format of comma-separated list (e.g., 0,3,1,0).
        lora_uids (list):
            The uids of LoRA weights for the input batch. Use -1 to disable the LoRA module.
        streaming (bool):
            Whether or not to use streaming mode for generation.
        stopping_criteria (StoppingCriteria):
            Custom stopping criteria.
        logits_processor (LogitsProcessor):
            Custom logits processors.
        kwargs (Dict[str, Any]:
            Ad hoc parametrization of sampling_config.
            The passed **kwargs matching the sampling_config's attributes will override them.
    Returns:
        torch.Tensor or dict:
            If return_dict=False, the method returns generated output_ids.
            If return_dict=True, the method returns a dict of output_ids,
            sequence_lengths (if sampling_config.output_sequence_lengths=True),
            context_logits and generation_logits (if self.gather_context_logits=True and
            self.gather_generation_logits=True, respectively).
    """
    assert streaming
    # TODO: Check if these can be supported now and support them
    if lora_uids is not None:
        raise RuntimeError("LoRA is not supported in C++ session.")
    if stopping_criteria is not None:
        raise RuntimeError("Stopping criteria is not supported in C++ session.")
    if logits_processor is not None:
        raise RuntimeError("Logits processor is not supported in C++ session.")

    # If we are in a multi-gpu scenario, only rank 0 continues
    if not runner.session.can_enqueue_requests():
        return []

    # Convert tensor input to plain lists
    batch_input_ids_list = [a.tolist() for a in batch_input_ids]

    if sampling_config is None:
        # Convert from old API of SamplingConfig
        # Note: Due to a Python3.10 bug one cannot use inspect on it currently
        accepted_parameters = [
            "num_beams",
            "top_k",
            "top_p",
            "top_p_min",
            "top_p_reset_ids",
            "top_p_decay",
            "random_seed",
            "temperature",
            "min_length",
            "beam_search_diversity_rate",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "length_penalty",
            "early_stopping",
        ]
        rename_params = {"num_beams": "beam_width"}
        sampling_params = {k: v for k, v in kwargs.items() if k in accepted_parameters}
        for k, v in rename_params.items():
            if k in sampling_params:
                sampling_params[v] = sampling_params.pop(k)
        if "top_p" in sampling_params and sampling_params["top_p"] == 0.0:
            sampling_params["top_p"] = None

        # To prevent numerical overflow when the temperature is set to 0.0
        # Attributes of `trtllm.SamplingConfig` cannot be modified
        if "temperature" in sampling_params and sampling_params["temperature"] == 0.0:
            sampling_params["temperature"] = None
            sampling_params["top_k"] = 1

        sampling_config = trtllm.SamplingConfig(**sampling_params)
    else:
        sampling_config = copy.deepcopy(sampling_config)

        # To prevent numerical overflow when the temperature is set to 0.0
        # Modify generation.SamplingConfig
        if isinstance(sampling_config.temperature, float) and sampling_config.temperature == 0.0:
            sampling_config.temperature = None
            sampling_config.top_k = 1

    runner._check_inputs(batch_input_ids_list, sampling_config, max_new_tokens)

    output_config = trtllm.OutputConfig(
        return_context_logits=runner.gather_context_logits,
        return_generation_logits=runner.gather_generation_logits,
        return_log_probs=output_log_probs,
    )

    prompt_tuning_configs = len(batch_input_ids_list) * [None]
    if prompt_table is not None:
        prompt_table_data = runner._prepare_embedding_table(prompt_table)
        if prompt_tasks is not None:
            task_indices = [int(t) for t in prompt_tasks.split(",")]
            assert len(task_indices) == len(
                batch_input_ids_list
            ), f"Number of supplied tasks ({len(task_indices)}) must match input batch size ({len(batch_input_ids_list)})"
            prompt_tuning_configs = [
                trtllm.PromptTuningConfig(embedding_table=prompt_table_data[task_indices[i]])
                for i in range(len(batch_input_ids_list))
            ]
        else:
            prompt_tuning_configs = [
                trtllm.PromptTuningConfig(embedding_table=prompt_table_data[0])
                for _ in range(len(batch_input_ids_list))
            ]

    requests = [
        trtllm.Request(
            input_token_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pad_id=pad_id,
            end_id=end_id,
            # not letting trtllm handle stop words as this is only supported on a token-level
            stop_words=None,
            bad_words=bad_words_list,
            sampling_config=sampling_config,
            streaming=streaming,
            output_config=output_config,
            prompt_tuning_config=prompt_tuning_configs[i],
        )
        for i, input_ids in enumerate(batch_input_ids_list)
    ]

    request_ids = runner.session.enqueue_requests(requests)
    multi_responses = runner.session.await_responses(request_ids)

    output_ids = [[] for _ in range(len(multi_responses))]
    for responses in multi_responses:
        for response in responses:
            if not response.has_error():
                reqid_pos = request_ids.index(response.request_id)
                if not streaming:
                    output_ids[reqid_pos] = [[] for _ in range(len(response.result.output_token_ids))]
                else:
                    output_ids[reqid_pos] = [
                        copy.deepcopy(batch_input_ids_list[reqid_pos])
                        for _ in range(len(response.result.output_token_ids))
                    ]

    return _stream(
        runner,
        request_ids,
        output_ids,
        multi_responses,
        end_id,
        return_dict,
        output_sequence_lengths,
        output_log_probs,
        output_cum_log_probs,
        batch_input_ids,
        streaming,
        stop_words_list,
        tokenizer,
        input_lengths,
    )


def _stream(
    runner,
    request_ids,
    output_ids,
    multi_responses,
    end_id,
    return_dict,
    output_sequence_lengths,
    output_log_probs,
    output_cum_log_probs,
    batch_input_ids,
    streaming,
    stop_words_list,
    tokenizer,
    input_lengths,
):
    if stop_words_list is None:
        stop_words_list = []
    active_reqids = copy.deepcopy(request_ids)
    assert len(active_reqids) == 1

    # checking the last 20 tokens for stop words
    num_tokens_to_check = 20

    idx = 0
    while active_reqids:
        for req_id, response in zip(active_reqids, multi_responses):
            for r in response:
                if r.result.is_final:
                    active_reqids.remove(req_id)

            output_ids = runner._process_response(multi_responses, output_ids, request_ids)
            output = runner._fill_output(
                multi_responses,
                output_ids,
                end_id,
                return_dict,
                output_sequence_lengths,
                output_log_probs,
                output_cum_log_probs,
                batch_input_ids,
                streaming,
            )

            matching_stop_word = None
            # checking every half of the required tokens to have overlapping checks
            if idx < num_tokens_to_check - 1 or idx % (num_tokens_to_check // 2) != 0:
                continue
            seq_length = output["sequence_lengths"]
            generation_suffix = output["output_ids"][0, 0, seq_length[0] - num_tokens_to_check : seq_length[0]]
            output_string = get_output_single(generation_suffix, 0, num_tokens_to_check, tokenizer, end_id)
            for stop_word in stop_words_list:
                if stop_word in output_string:
                    matching_stop_word = stop_word
                    break

            if matching_stop_word is not None:
                runner.session.cancel_request(req_id)
                if req_id in active_reqids:
                    active_reqids.remove(req_id)
                break

        if active_reqids:
            multi_responses = runner.session.await_responses(active_reqids)
        idx += 1

    output_string = get_output(output["output_ids"], input_lengths, output["sequence_lengths"][0], tokenizer, end_id)[
        0
    ]
    for stop_word in stop_words_list:
        if stop_word in output_string:
            matching_stop_word = stop_word
            break
    output = {
        "output_ids": output["output_ids"][0, 0].tolist(),
    }
    ends_properly = False
    if matching_stop_word is not None:
        output_string = remove_stop_tokens(output_string, stop_words_list)
        beg = seq_length[0] - num_tokens_to_check
        end = seq_length[0]
        # find the token ids that stops at the stop word
        for i in range(beg, end):
            if tokenizer.decode(output["output_ids"][:i]).endswith(stop_word):
                output["output_ids"] = output["output_ids"][:i]
                break
        # adding it back, since we only need to remove what's *after* the stop phrase
        output_string += matching_stop_word
        ends_properly = True
    output["output_ids"] = output["output_ids"][input_lengths[0] :]
    output["generation"] = output_string
    output["ends_properly"] = ends_properly
    return output


class ExtendedTrtStartGeneration(TrtStartGeneration):
    def start_generation(
        self,
        prompt,
        input_ids,
        input_length,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        random_seed,
        stop_words_list,
    ):
        return self.model.start_generation(
            prompt=prompt,
            input_ids=input_ids,
            input_length=input_length,
            max_output_token=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_words_list=stop_words_list,
        )

    def put(self):
        logging.debug("generate async request")
        logging.debug("request IP: %s", str(request.remote_addr))
        input_request = request.get_json()
        logging.debug("request content: %s", json.dumps(input_request))

        top_k = input_request.get("top_k")
        if top_k == 0:
            top_k = None
        data = dict(
            prompt=input_request.get("prompt", None),
            input_ids=input_request.get("input_ids"),
            input_length=input_request.get("input_length"),
            max_new_tokens=input_request.get("tokens_to_generate", 64),
            temperature=input_request.get("temperature", 1.0),
            top_k=top_k,
            top_p=input_request.get("top_p", 1.0),
            repetition_penalty=input_request.get("repetition_penalty", 1.2),
            random_seed=input_request.get("random_seed", 0),
            stop_words_list=input_request.get("stop_words_list"),
        )
        self.comm.Barrier()
        data = self.comm.bcast(data, root=0)

        out = self.start_generation(**data)
        return jsonify(out)


class ExtendedTensorRTLLM(TensorRTLLM):
    def get_output(
        self,
        batch_input_ids,
        input_lengths,
        max_output_token,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        random_seed,
        stop_words_list,
    ):
        # TODO: return dictionary with a proper error reporting

        try:
            output = generate(
                self.runner,
                batch_input_ids,
                input_lengths,
                max_new_tokens=max_output_token,
                end_id=self.end_id,
                pad_id=self.pad_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                random_seed=random_seed,
                # stop words in trtllm are supported on the token-level only and this representation is not unique
                # so instead of passing in all tokenizations (is that even possible?) of each phrase, we will
                # instead stream outputs and detokenize them to check for stop words - this is done inside
                # overriden generate/stream functions above
                tokenizer=self.tokenizer,
                stop_words_list=stop_words_list,
                return_dict=True,
                output_sequence_lengths=True,
                streaming=True,
            )
        except RuntimeError as e:
            logging.error("RuntimeError: %s", e)
            output = f"RuntimeError: {e}"

        return output

    def get_result(self, idx):
        if self.requests[idx].done():
            result = self.requests.pop(idx).result()
            return result
        return None

    @torch.no_grad()
    def start_generation(
        self,
        prompt,
        input_ids,
        input_length,
        max_output_token,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        random_seed,
        stop_words_list,
    ):
        idx = str(uuid.uuid4())
        if prompt is not None:
            batch_input_ids, input_lengths = parse_input([prompt], self.tokenizer)
            batch_input_ids = batch_input_ids[0]
        else:
            batch_input_ids = torch.tensor([input_ids])
            input_lengths = [input_length]
        self.requests[idx] = self.executor.submit(
            self.get_output,
            batch_input_ids,
            input_lengths,
            max_output_token,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            random_seed,
            stop_words_list,
        )

        return idx


class TRTServer:
    def __init__(self, model_path: str):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self.model = ExtendedTensorRTLLM(model_path=model_path)

        if self.rank == 0:
            self.app = Flask(__file__, static_url_path="")
            api = Api(self.app)
            api.add_resource(ExtendedTrtStartGeneration, "/start_generation", resource_class_args=[self.model])
            api.add_resource(TrtGetResult, "/get_result", resource_class_args=[self.model])

    def run(self, url, port=5000):
        if self.rank == 0:
            self.app.run(url, threaded=True, port=port, debug=False)
        else:
            self.worker_loop()

    def worker_loop(self):
        server = ExtendedTrtStartGeneration(self.model)
        while True:
            self.comm.Barrier()
            data = None
            data = self.comm.bcast(data, root=0)
            server.start_generation(**data)


class TensorRTLLMModelClient(TensorRTLLMModel):
    """Note that the current implementation supports inflight-batching so
    to make the most use of it, you should submit a large number of prompts
    at the same time.

    A good default value is 16-32 times bigger than the model's max batch size.
    """

    def generate(
        self,
        prompts: list[str] = None,
        batch_input_ids: list[list[int]] = None,
        input_lengths: list[int] = None,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
        if stop_phrases is None:
            stop_phrases = []
        request = {
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "stop_words_list": stop_phrases,
        }
        preprocess_request(request)

        generation_ids = []

        large_prime = 2147462143
        if prompts is not None:
            for i, prompt in enumerate(prompts):
                request["prompt"] = prompt
                modified_random = hash(str(random_seed) + str(i)) % large_prime
                request['random_seed'] = modified_random
                generation_ids.append(
                    self.requests_lib.put(
                        url="http://{}:{}/start_generation".format(self.server_host, self.server_port),
                        data=json.dumps(request),
                        headers={"Content-Type": "application/json"},
                    ).json()
                )
        else:
            for i, (input_ids, lengths) in enumerate(zip(batch_input_ids, input_lengths)):
                request["input_ids"] = input_ids
                request["input_length"] = lengths
                modified_random = hash(str(random_seed) + str(i)) % large_prime
                request['random_seed'] = modified_random
                generation_ids.append(
                    self.requests_lib.put(
                        url="http://{}:{}/start_generation".format(self.server_host, self.server_port),
                        data=json.dumps(request),
                        headers={"Content-Type": "application/json"},
                    ).json()
                )

        outputs = [None] * len(generation_ids)
        finished_count = 0
        while finished_count < len(generation_ids):
            time.sleep(0.1)
            for pos, generation_id in enumerate(generation_ids):
                if outputs[pos] is not None:
                    continue
                result = self.requests_lib.put(
                    url="http://{}:{}/get_result".format(self.server_host, self.server_port),
                    data=json.dumps({"generation_id": generation_id}),
                    headers={"Content-Type": "application/json"},
                ).json()
                if result is not None:
                    finished_count += 1
                    outputs[pos] = result
        if remove_stop_phrases:
            postprocess_output(outputs, stop_phrases)
        return outputs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    server = TRTServer(model_path=args.model_path)
    server.run(args.host, args.port)
