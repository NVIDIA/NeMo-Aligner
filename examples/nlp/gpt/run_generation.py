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
import os
from functools import partial

import jsonlines
import torch
import torch.multiprocessing as mp
from megatron.core.utils import divide
from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.code_execution.sandbox import get_sandbox
from omegaconf.omegaconf import OmegaConf
from tqdm import tqdm

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.grpo import GRPOTrainer
from nemo_aligner.data.nlp.builders import build_dataloader, collate_with_pad_to_max_batch
from nemo_aligner.models.nlp.gpt.megatron_gpt_ppo_actor import MegatronGPTActorModel
from nemo_aligner.models.nlp.gpt.reward_critic_clients import RemoteGPTRMCriticClient
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.batch_iterators import get_batch_iterator_cls
from nemo_aligner.utils.distributed import Timer, run_if_model_parallel_src
from nemo_aligner.utils.parallel_state import get_model_parallel_group, get_model_parallel_src_rank
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_peft,
    init_using_ptl,
    resolve_and_create_trainer,
    retrieve_custom_trainer_state_dict,
)
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, retrieve_model_state_dict_in_cpu

"""Script to start PPO training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)
OmegaConf.register_new_resolver("subtract", lambda x, y: x - y, replace=True)

mp.set_start_method("spawn", force=True)

PROMPT_TEMPLATE = """<extra_id_0>System

<extra_id_1>User
Below is a math question. I want you to first reason through the steps required to reach the answer, then put the answer (and only answer) inside \\boxed{{}}. For instance, if the answer is 42 then your response must end with \\boxed{{42}}.
{problem}
<extra_id_1>Assistant
"""


class MathDataset:
    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer

        assert os.path.exists(self.data_path), f"{self.data_path} must exist"

        with jsonlines.open(self.data_path) as reader:
            self.data = [obj for obj in reader]

    def __len__(self):
        return len(self.data)

    def encode(self, text):
        text_ids = self.tokenizer.text_to_ids(text)
        return text_ids, len(text_ids)

    def __getitem__(self, idx):
        """
        Return a single prompt.
        """
        text = PROMPT_TEMPLATE.format(problem=self.data[idx]["problem"])
        answer = self.data[idx]["expected_answer"]

        sample, _ = self.encode(text)
        sample_tensor = torch.as_tensor(sample, dtype=torch.int64)

        output = {
            "text": sample_tensor,
            "length": sample_tensor.shape[0],
            "answer": answer,
            "loss_multiplier": True,
            "idx": idx,
        }
        return output


@hydra_runner(config_path="conf", config_name="gpt_grpo_actor")
def main(cfg) -> None:
    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "ppo")

    exp_manager(trainer, cfg.exp_manager)

    logger = CustomLoggerWrapper(trainer.loggers)

    ptl_model = load_from_nemo(
        MegatronGPTActorModel,
        cfg.model,
        trainer,
        strict=True,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )
    ptl_model.freeze()

    init_peft(ptl_model, cfg.model)
    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    ptl_model = ptl_model.cuda()
    ptl_model.prepare_for_inference()

    validation_ds = MathDataset(cfg.model.data.data_prefix["validation"][0], ptl_model.tokenizer)

    max_seqlen = cfg.model.ppo.length_params.max_length
    eos_id = ptl_model.tokenizer.eos_id

    # collate fn to pad to the max seq length in the batch
    collate_fn = collate_with_pad_to_max_batch(max_seqlen, eos_id, cfg, generate_masks_and_position_ids=False)

    val_dataloader_builder = partial(
        build_dataloader,
        cfg=cfg,
        dataset=validation_ds,
        mbs=cfg.model.ppo.rollout_micro_batch_size,
        gbs=cfg.model.ppo.rollout_micro_batch_size * parallel_state.get_data_parallel_world_size(),
        collate_fn=collate_fn,
        load_gbs=True,
        use_random_sampler=False,
    )

    logger.log_hyperparams(OmegaConf.to_container(cfg))

    sandbox = get_sandbox()

    def sandbox_call(answers):
        return [sandbox.is_output_correct(*item) for item in answers]

    outputs = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader_builder(consumed_samples=0)):
            rollout_batch = ptl_model.infer(batch, use_greedy=True)

            texts = [
                ptl_model.tokenizer.ids_to_text(item[length:].tolist())
                for item, length in zip(rollout_batch["response_tokens"], batch["length"])
            ]
            answers = [(extract_answer(t), a) for t, a in zip(texts, batch["answers"])]

            answer = run_if_model_parallel_src(sandbox_call, answers)

            prompts = [
                item[:length]
                for item, length in zip(
                    ptl_model.tokenizer.ids_to_text(batch["text"].tolist()), batch["length"].tolist()
                )
            ]

            prompts = ptl_model.tokenizer.ids_to_text(batch["text"].tolist())

            src_rank = get_model_parallel_src_rank()
            if torch.distributed.get_rank() == src_rank:

                for a, resp, prompt, idx in zip(answer, texts, prompts, batch["idx"]):
                    outputs.append({"response": resp, "answer": a, "prompt:": prompt, **validation_ds.data[idx]})

    src_rank = get_model_parallel_src_rank()
    if torch.distributed.get_rank() == src_rank:
        save_path = os.path.join(
            cfg.exp_manager.explicit_log_dir, "rank_{}_results.jsonl".format(parallel_state.get_data_parallel_rank())
        )

        with jsonlines.open(save_path, mode="w") as writer:
            writer.write_all(outputs)


if __name__ == "__main__":
    main()
