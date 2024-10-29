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

from collections import defaultdict
from statistics import mean

import torch
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingRandomBatchSampler,
)
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.utils import logging
from nemo_aligner.algorithms.dpo import DPOTrainer
from nemo_aligner.utils.distributed import SyncTimer
from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.trainer_utils import check_progress, compute_limit_batches
from nemo_aligner.utils.utils import clear_memory


def kto_custom_collate(batch, eos_id, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False):
    # sample_tokens = [item["sample"] for item in batch]
    sample_tokens = [torch.cat((item["prompt_tokens"], item["response_tokens"]), dim=0) for item in batch]
    sample_lengths = torch.LongTensor([item["sample_length"] for item in batch])
    sample_labels = [item["sample_labels"] for item in batch]
    sample_preference = torch.tensor([item["preference"] for item in batch])

    batch_size = len(batch)
    # We estimate the KL divergence term from non-matching prompt-response pairs in the batch. For that purpose,
    # we build samples by combining the every prompt in the batch with the reponse of the subsequent sample
    indices = list(range(1, batch_size)) + [0]
    kl_sample_tokens = [
        torch.cat((item["prompt_tokens"], batch[indices[k]]["response_tokens"]), dim=0) for k, item in enumerate(batch)
    ]
    kl_sample_labels = [
        torch.cat(
            (
                -100 * torch.ones(item["prompt_tokens"].size(0), dtype=torch.long),
                kl_sample_tokens[k][item["prompt_tokens"].size(0) :],
            )
        )
        for k, item in enumerate(batch)
    ]

    all_tokens = sample_tokens + kl_sample_tokens
    all_labels = sample_labels + kl_sample_labels

    all_tokens = torch.nn.utils.rnn.pad_sequence(all_tokens, batch_first=True, padding_value=eos_id)
    all_labels = torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=-100)

    sample_tokens = all_tokens[:batch_size]
    sample_labels = all_labels[:batch_size]

    kl_sample_tokens = all_tokens[batch_size:]
    kl_sample_labels = all_labels[batch_size:]

    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        sample_tokens, eos_id, reset_position_ids, reset_attention_mask, eod_mask_loss,
    )
    assert attention_mask.ndim == 4, "attention_mask is incorrect shape for dpo_custom_collate"
    if attention_mask.shape[0] == 1:
        # using .expand() here causes errors from pin_memory=True, so need to use .repeat()
        # attention_mask = attention_mask.expand(len(batch), *((-1,) * (len(attention_mask.shape) - 1)))
        attention_mask = attention_mask.repeat(len(batch), *((1,) * (len(attention_mask.shape) - 1)))

    output = {
        "samples": sample_tokens,
        "kl_samples": kl_sample_tokens,
        "sample_length": sample_lengths,
        "sample_labels": sample_labels,
        "kl_sample_labels": kl_sample_labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "preference": sample_preference,
    }
    return output


class KTOTrainer(DPOTrainer):
    """Trainer to coordinate KTO training
    """

    def __init__(
        self,
        cfg: DictConfig,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        logger,
        ckpt_callback,
        run_timer,
    ):
        super().__init__(
            cfg,
            model,
            optimizer,
            scheduler,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            logger,
            ckpt_callback,
            run_timer,
        )

    def augment_dataloader(self, dataloader):
        """Augment dataloader with ref policy log prob"""
        iter_dataloader = iter(dataloader)
        while True:
            try:
                batch = next(iter_dataloader)
                logprobs = self.model.get_ref_policy_logprobs(batch).cpu()
                samples_logps, kl_samples_logps = torch.split(logprobs, len(logprobs) // 2, dim=0)
                batch["ref_policy_log_probs_samples"] = samples_logps
                batch["ref_policy_log_probs_kl_samples"] = kl_samples_logps

                yield batch
                del logprobs, samples_logps, kl_samples_logps
            except StopIteration:
                break
