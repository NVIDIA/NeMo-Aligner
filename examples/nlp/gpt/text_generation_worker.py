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

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
import datetime
import os
import random
import time

import torch
from celery import Celery
from datasets import load_dataset
from omegaconf import OmegaConf, open_dict
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.modules.common.text_generation_strategy import model_inference_strategy_dispatcher
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import CustomProgressBar, NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils import AppState, logging
from nemo.utils.model_utils import inject_model_parallel_rank
from nemo_aligner.utils.distributed import Timer, broadcast_python_obj
from nemo_aligner.utils.train_script_utils import resolve_and_create_trainer

"""Script to start Reward Model training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)
OmegaConf.register_new_resolver("not", lambda x: not x)


def init_distributed_parameters(trainer, cfg):
    app_state = AppState()
    if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
        app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
        app_state.tensor_model_parallel_size = cfg.tensor_model_parallel_size
        app_state.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
        (
            app_state.tensor_model_parallel_rank,
            app_state.pipeline_model_parallel_rank,
            app_state.model_parallel_size,
            app_state.data_parallel_size,
            app_state.pipeline_model_parallel_split_rank,
            app_state.virtual_pipeline_model_parallel_rank,
        ) = fake_initialize_model_parallel(
            world_size=app_state.model_parallel_size,
            rank=trainer.global_rank,
            tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
            pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
        )
    if trainer.num_devices and trainer.num_nodes:
        app_state.world_size = trainer.num_devices * trainer.num_nodes


@hydra_runner(config_path="conf", config_name="gpt_inference")
def main(cfg) -> None:
    logging.info("\n\n************** Inference configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    # trainer required for restoring model parallel models
    # trainer required for restoring model parallel models
    trainer = Trainer(
        strategy=NLPDDPStrategy(timeout=datetime.timedelta(seconds=18000)),
        **cfg.trainer,
        callbacks=[CustomProgressBar()],
    )

    if cfg.gpt_model_file is not None:
        if (
            cfg.tensor_model_parallel_size < 0
            or cfg.pipeline_model_parallel_size < 0
            or cfg.get("pipeline_model_parallel_split_rank", -1) < 0
        ):
            save_restore_connector = NLPSaveRestoreConnector()
            if os.path.isdir(cfg.gpt_model_file):
                save_restore_connector.model_extracted_dir = cfg.gpt_model_file
            model_config = MegatronGPTModel.restore_from(
                restore_path=cfg.gpt_model_file,
                trainer=trainer,
                return_config=True,
                save_restore_connector=save_restore_connector,
            )

            # with dist checkpointing we don't need to set this
            if not model_config.get("mcore_gpt", False):
                with open_dict(cfg):
                    cfg.tensor_model_parallel_size = model_config.get("tensor_model_parallel_size", 1)
                    cfg.pipeline_model_parallel_size = model_config.get("pipeline_model_parallel_size", 1)
                    cfg.pipeline_model_parallel_split_rank = model_config.get("pipeline_model_parallel_split_rank", 0)

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    if cfg.gpt_model_file:
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.gpt_model_file):
            save_restore_connector.model_extracted_dir = cfg.gpt_model_file

        pretrained_cfg = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        OmegaConf.set_struct(pretrained_cfg, True)
        with open_dict(pretrained_cfg):
            pretrained_cfg.sequence_parallel = False
            pretrained_cfg.activations_checkpoint_granularity = None
            pretrained_cfg.activations_checkpoint_method = None
            pretrained_cfg.precision = trainer.precision
            if pretrained_cfg.get("mcore_gpt", False):
                # with dist checkpointing we can use the model parallel config specified by the user
                pretrained_cfg.tensor_model_parallel_size = cfg.tensor_model_parallel_size
                pretrained_cfg.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
            if trainer.precision == "16":
                pretrained_cfg.megatron_amp_O2 = False
            elif trainer.precision in ["bf16", "bf16-mixed"] and cfg.get("megatron_amp_O2", False):
                pretrained_cfg.megatron_amp_O2 = True
        model = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=save_restore_connector,
            map_location=f"cuda:{trainer.local_rank}",  # map_location is needed for converted models
        )
    elif cfg.checkpoint_dir:
        # checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
        # trainer._checkpoint_connector.restore(checkpoint_path)
        # trainer._checkpoint_connector._restore_modules_and_callbacks(cfg.checkpoint_dir)
        init_distributed_parameters(trainer, cfg)
        checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
        # checkpoint_path is a dir in case of distributed checkpointing
        if not os.path.isdir(checkpoint_path):
            # legacy checkpoint needs model parallel rank injection
            checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
        # overwrite parameters
        overwrite_cfg = {
            "sequence_parallel": False,
            "activations_checkpoint_granularity": None,
            "activations_checkpoint_method": None,
            "precision": trainer.precision,
            "tensor_model_parallel_size": cfg.tensor_model_parallel_size,
            "pipeline_model_parallel_size": cfg.pipeline_model_parallel_size,
            "megatron_amp_O2": cfg.get("megatron_amp_O2", False),
            "tokenizer": cfg.tokenizer,
        }
        model = MegatronGPTModel.load_from_checkpoint(
            checkpoint_path, hparams_file=None, trainer=trainer, **overwrite_cfg
        )
        # model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer)
        # # the world_size is set to 1, because it read from trainer.world_size when loading the checkponts
        # init_distributed_parameters(trainer, cfg)
    else:
        raise ValueError("need at least a nemo file or checkpoint dir")

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    fp8_enabled = hasattr(model.cfg, "fp8") and (model.cfg.fp8 == True)
    if fp8_enabled:
        raise ValueError("fp8 is not supported for inference")

    # start the worker on the rank
    start_worker(model, cfg, cfg.server_url, cfg.backend_url)


def start_worker(model, cfg, url, backend_url):
    if torch.distributed.get_rank() == 0:
        app = Celery("tasks", backend=f"{backend_url}", broker=f"{url}")

        app.conf.task_acks_late = True
        app.conf.worker_deduplicate_successful_tasks = True
        app.conf.worker_prefetch_multiplier = 1

        # 5 hrs timeout
        app.conf.update(broker_transport_options={"visibility_timeout": 18000},)

        def process_input(batch, cfg):
            length_params: LengthParam = {
                "max_length": cfg.inference.tokens_to_generate,
                "min_length": cfg.inference.min_tokens_to_generate,
            }

            sampling_params: SamplingParam = {
                "use_greedy": cfg.inference.greedy,
                "temperature": cfg.inference.temperature,
                "top_k": cfg.inference.top_k,
                "top_p": cfg.inference.top_p,
                "repetition_penalty": cfg.inference.repetition_penalty,
                "add_BOS": cfg.inference.add_BOS,
                "all_probs": cfg.inference.all_probs,
                "compute_logprob": cfg.inference.compute_logprob,
                "end_strings": cfg.inference.end_strings,
            }

            length_params.update(batch["length_params"])
            sampling_params.update(batch["sampling_params"])
            inputs = batch["inputs"]
            return {"inputs": inputs, "length_params": length_params, "sampling_params": sampling_params}

        @app.task(
            bind=True,
            autoretry_for=(Exception,),
            retry_backoff=True,
            retry_jitter=True,
            retry_kwargs={"max_retries": 10},
        )
        def text_generation_batch(self, batch):
            try:
                beg = time.time()
                batch = broadcast_python_obj(batch, 0, None)
                args = process_input(batch, cfg)
                inputs = args["inputs"]
                length_params = args["length_params"]
                sampling_params = args["sampling_params"]
                response = generate(
                    model,
                    inputs=inputs,
                    tokens_to_generate=length_params["max_length"],
                    all_probs=sampling_params["all_probs"],
                    compute_logprob=sampling_params["compute_logprob"],
                    temperature=sampling_params["temperature"],
                    add_BOS=sampling_params["add_BOS"],
                    top_k=sampling_params["top_k"],
                    top_p=sampling_params["top_p"],
                    greedy=sampling_params["use_greedy"],
                    repetition_penalty=sampling_params["repetition_penalty"],
                    end_strings=sampling_params["end_strings"],
                    min_tokens_to_generate=length_params["min_length"],
                    compute_attention_mask=sampling_params.get("compute_attention_mask", True),
                )
                inference_strategy = model_inference_strategy_dispatcher(model)
                context_tokens_tensor, context_length_tensor = inference_strategy.tokenize_batch(
                    inputs, length_params["max_length"], sampling_params["add_BOS"]
                )

                output_response = [
                    model.tokenizer.ids_to_text(tokens[length.item() : length.item() + length_params["max_length"]])
                    for tokens, length in zip(response["token_ids"], context_length_tensor)
                ]
                # remove the end_strings from the output
                clean_output_response = []
                for clean_response in output_response:
                    for end_string in sampling_params["end_strings"]:
                        if clean_response.endswith(end_string):
                            clean_response = clean_response[: -len(end_string)]
                            clean_response = clean_response.strip()
                    clean_output_response.append(clean_response)
                output = {}
                output["responses"] = clean_output_response

                if "logprob" in response and response["logprob"] is not None:
                    eod_id = model.tokenizer.eos_id
                    # log probs need to offset by 1 because the first token has no log prob
                    logprob_context_length = context_length_tensor - 1
                    logprob_list = []
                    for tokens, logprob, context_len in zip(
                        response["token_ids"], response["logprob"], logprob_context_length
                    ):
                        context_len = context_len.item()
                        tokens = torch.tensor(tokens[1:])
                        logprobs = logprob[context_len : context_len + length_params["max_length"]]
                        tokens = tokens[context_len : context_len + length_params["max_length"]]
                        all_logprob = logprobs[tokens != eod_id].sum().item()
                        logprob_list.append(all_logprob)
                    output["response_logprob"] = logprob_list
                    output["logprob"] = response["logprob"].tolist()
                output["context_length"] = context_length_tensor.tolist()
                output["token_ids"] = response["token_ids"]
                if "full_logprob" in response and response["full_logprob"] is not None:
                    output["full_logprob"] = response["full_logprob"].tolist()
                # compute the response logprob
                if "data_ids" in batch:
                    output["data_ids"] = batch["data_ids"]
                output["time"] = time.time() - beg
            except Exception as e:
                print("ERROR", e)
                import traceback

                traceback.print_exc()
                raise self.retry(exc=e)
            return output

        RAND_ID = os.environ.get("local_rank", random.randint(0, 10000))
        app.worker_main(
            [
                "worker",
                "--loglevel=INFO",
                "--concurrency=1",
                "--pool=threads",
                "--without-gossip",
                "-Ofair",
                f"--hostname=worker-{RAND_ID}@%%h",
            ]
        )
    else:
        while True:
            batch = None
            batch = broadcast_python_obj(batch, 0, None)
            args = process_input(batch, cfg)
            # braodcast the
            # First method of running text generation, call model.generate method
            generate(model)


if __name__ == "__main__":
    main()
