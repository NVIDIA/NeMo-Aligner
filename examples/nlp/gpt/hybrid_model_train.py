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
from collections import defaultdict
from copy import deepcopy
from functools import partial

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from megatron.core import parallel_state
from megatron.core.utils import divide
from omegaconf.omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.deepsearch import DeepSearchTrainer
from nemo_aligner.data.nlp.builders import build_dataloader
from nemo_aligner.models.nlp.gpt.megatron_gpt_hybrid_model import MegatronGPTHybridModel
from nemo_aligner.utils.deep_search.mcts.feedback_functions import GSK8KFeedbackDataset
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_using_ptl,
    resolve_and_create_trainer,
    retrieve_custom_trainer_state_dict,
)
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo

"""Script to start Reward Model training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)

steerlm_template = """<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<extra_id_1>User
{prompt}
Please show the calculation steps and lastly the final answer in format {{{{answer number}}}}
<extra_id_1>Assistant
<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2
"""


def collate_fn(batch):
    # applies the steerlm format and
    # transposes the list of dict to dict of lists
    new_dict = defaultdict(list)

    for b in batch:
        new_dict["question"].append(steerlm_template.format(prompt=b["question"]))
        new_dict["answer"].append(b["answer"])
        new_dict["data_id"].append(b["data_id"])

    return new_dict


class DatasetWrapper:
    def __init__(self, ds):
        self.ds = ds

    # just like a dataset but return idx
    def __getitem__(self, idx):
        return {**self.ds[idx], "data_id": idx}

    def __len__(self):
        return len(self.ds)


def fill_padded_tensor_with_data(batches, max_seqlen, lengths, response_lengths, pad_value):
    """unpadded x * -> B x max seq len x *"""
    assert len(batches) == len(lengths)

    output = batches[0].new_empty((len(batches), max_seqlen, *batches[0].shape[1:])).fill_(pad_value)

    for i, (batch, length, response_length) in enumerate(zip(batches, lengths, response_lengths, strict=True)):
        idx = length - 1
        output[i, idx:response_length, ...] = batch

    return output


def create_mask(tokens, lengths, response_length):
    idx = lengths - 1

    end = (response_length).view(-1, 1)
    start = idx.view(-1, 1)

    seq_range = torch.arange(tokens.size(-1), device=lengths.device).view(1, -1)
    sequence_mask = (start <= seq_range) & (end > seq_range)
    return sequence_mask


def mcts_collate_fn(eos_id, batch):
    new_dict = {}
    context_keys = {"context_length", "response_length"}
    token_keys = {"tokens"}
    fill_keys = {"action_probs", "reward", "actions"}

    max_seqlen = max(len(x["tokens"]) for x in batch)
    lengths = [x["context_length"] for x in batch]

    for k in context_keys | token_keys:
        batches = tuple(torch.as_tensor(x[k]) for x in batch)

        if k in context_keys:
            output = torch.stack(batches)
        elif k in token_keys:
            output = torch.nn.utils.rnn.pad_sequence(batches, batch_first=True, padding_value=eos_id,)

        new_dict[k] = output

    max_seqlen = new_dict["tokens"].size(-1)
    lengths = new_dict["context_length"]
    response_length = new_dict["response_length"]

    for k in fill_keys:
        output = fill_padded_tensor_with_data(
            tuple(torch.as_tensor(x[k]) for x in batch), max_seqlen, lengths, response_length, 0
        )
        new_dict[k] = output

    mask = create_mask(new_dict["tokens"], lengths, new_dict["response_length"])

    return new_dict | {"mcts_mask": mask}


def mcts_value_collate_fn(eos_id, batches):
    new_dict = defaultdict(list)

    for batch in batches:
        new_dict["tokens"].extend(batch["tokens"])
        new_dict["reward"].extend(batch["reward"])
        new_dict["response_length"].extend(list(len(x) for x in batch["tokens"]))
        new_dict["context_length"].extend([batch["context_length"]] * len(batch["tokens"]))

    final_dict = {}
    for k, v in new_dict.items():
        if k == "tokens":
            inputs = tuple(torch.as_tensor(x) for x in v)
            output = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=eos_id,)
        else:
            output = torch.as_tensor(v)

        final_dict[k] = output

    mask = create_mask(final_dict["tokens"], final_dict["context_length"], final_dict["response_length"])

    return final_dict | {"mcts_mask": mask}


@hydra_runner(config_path="conf", config_name="gpt_hybrid_train")
def main(cfg) -> None:
    optim = deepcopy(cfg.model.optim)
    val_ds = DatasetWrapper(load_dataset("gsm8k", "main")["test"])

    train_ds = DatasetWrapper(load_dataset("gsm8k", "main")["train"])
    feedback = GSK8KFeedbackDataset()

    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)
    # hard reset the optim flag
    cfg.model.optim = optim

    cfg.model.value = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model.value)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "deep_search")

    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    hybrid_model_cls = MegatronGPTHybridModel

    ptl_model = load_from_nemo(
        hybrid_model_cls,
        cfg.model,
        trainer,
        strict=True,
        load_base_model_only=True,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )

    trainer_restore_path = trainer.ckpt_path
    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
        consumed_samples = custom_trainer_state_dict["consumed_samples"]
        consumed_samples_values = custom_trainer_state_dict["consumed_samples_values"]
    else:
        custom_trainer_state_dict = None
        consumed_samples = 0
        consumed_samples_values = 0

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    dp_size = parallel_state.get_data_parallel_world_size()

    assert os.path.exists(cfg.mcts_data_file)
    train_data = torch.load(cfg.mcts_data_file)

    # TODO(geshen): should we shuffle the data?
    policy_data, value_data = train_data["policies"], train_data["values"]

    num_questions_correct = 0

    for p in policy_data:
        if all(x > 0 for x in p["reward"]):
            num_questions_correct += 1

    data_metrics = {
        "num_questions_correct": num_questions_correct,
        "num_questions": len(policy_data),
        "accuracy": num_questions_correct / len(policy_data),
    }

    logging.info("Loaded search cached data at {} with metric {}".format(cfg.mcts_data_file, data_metrics))

    eos_id = ptl_model.tokenizer.eos_id

    # TODO(geshen): consumed samples need to be different for each of these 2 dataloaders
    # TODO(geshen): support multiple epochs
    train_policy_dataloader = build_dataloader(
        cfg=cfg,
        dataset=policy_data,
        consumed_samples=consumed_samples,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        load_gbs=True,
        collate_fn=partial(mcts_collate_fn, eos_id),
    )

    # TODO(geshen): can have different mbs
    train_value_dataloader = build_dataloader(
        cfg=cfg,
        dataset=value_data,
        consumed_samples=consumed_samples_values,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        load_gbs=True,
        collate_fn=partial(mcts_value_collate_fn, eos_id),
    )

    # hack to allow using all of the validation dataset
    val_dataloader = build_dataloader(
        cfg=cfg,
        dataset=val_ds,
        consumed_samples=0,
        mbs=cfg.model.inference.micro_batch_size,
        gbs=cfg.model.inference.micro_batch_size * dp_size,
        load_gbs=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=0,
        mbs=cfg.model.inference.micro_batch_size,
        gbs=cfg.model.inference.micro_batch_size * dp_size,
        load_gbs=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # TODO(geshen): set the optimizer steps properly, just like in PPO
    # TODO(geshen): better use constant LR here
    init_using_ptl(trainer, ptl_model, train_policy_dataloader, None)

    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)
    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))
    timer = Timer(cfg.exp_manager.get("max_time_per_run"))

    deep_search_trainer = DeepSearchTrainer(
        cfg=cfg.trainer.deep_search,
        model=ptl_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_policy_dataloader=train_policy_dataloader,
        train_value_dataloader=train_value_dataloader,
        val_dataloader=val_dataloader,
        train_dataloader=train_dataloader,
        feedback=feedback,
        logger=logger,
        ckpt_callback=ckpt_callback,
        run_timer=timer,
    )

    if custom_trainer_state_dict is not None:
        deep_search_trainer.load_state_dict(custom_trainer_state_dict)

    logger.log_metrics(data_metrics, step=deep_search_trainer.step, prefix="data/")

    deep_search_trainer.fit()


if __name__ == "__main__":
    main()
