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

from importlib.metadata import version
from itertools import chain
from typing import List, Optional, Tuple, Union

import hydra
import torch
from apex.transformer.pipeline_parallel.utils import get_micro_batch_size, get_num_microbatches
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf.dictconfig import DictConfig
from pkg_resources import packaging
from pytorch_lightning.trainer.trainer import Trainer
from torch.distributed import barrier

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
)
from nemo.collections.nlp.modules.common.text_generation_strategy import TextGenerationStrategy
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_default_length_params,
    get_default_sampling_params,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, OutputType, SamplingParam
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import logging
from nemo_aligner.models.alignable_interface import SupervisedInterface
from nemo_aligner.models.nlp.gpt.gpt_sft_model import GPTSFTModel
from nemo_aligner.utils.train_utils import (
    finish_validation_step,
    grad_reductions,
    prepare_for_training_step,
    prepare_for_validation_step,
    set_eval,
    set_sync_funcs,
    set_train,
)
from nemo_aligner.utils.utils import configure_batch_sizes

# def get_grouped_iterator_k_split(batch, num_microbatches):
#     microbatches = []
#     total_items = sum([len(group["tokens"]) for group in batch])
#     assert (
#         total_items % num_microbatches == 0
#     ), f"total number of items in the batch {total_items} must be divisible by num_microbatches {num_microbatches}"
#     assert (
#         num_microbatches % len(batch) == 0
#     ), f"num_microbatches {num_microbatches} must be divisible by the number of groups in the batch {len(batch)}"
#     sub_num_microbatches = num_microbatches // len(batch)
#     for group in batch:
#         if isinstance(group, dict):
#             items = list(group.items())
#             split_batch = [torch.tensor_split(item[1], sub_num_microbatches, dim=0) for item in items]
#             one_microbatches = [
#                 [(items[i][0], split_batch[i][j]) for i in range(len(items))] for j in range(sub_num_microbatches)
#             ]
#             one_microbatches = [dict(elem) for elem in one_microbatches]
#             microbatches.extend(one_microbatches)
#
#     return chain(microbatches)


def get_grouped_iterator_k_split(group, sub_num_microbatches):
    microbatches = []
    # total_items = sum([len(group["tokens"]) for group in batch])
    # assert (
    #     total_items % num_microbatches == 0
    # ), f"total number of items in the batch {total_items} must be divisible by num_microbatches {num_microbatches}"
    # assert (
    #     num_microbatches % len(batch) == 0
    # ), f"num_microbatches {num_microbatches} must be divisible by the number of groups in the batch {len(batch)}"
    if isinstance(group, dict):
        items = list(group.items())
        split_batch = [torch.tensor_split(item[1], sub_num_microbatches, dim=0) for item in items]
        one_microbatches = [
            [(items[i][0], split_batch[i][j]) for i in range(len(items))] for j in range(sub_num_microbatches)
        ]
        one_microbatches = [dict(elem) for elem in one_microbatches]
        microbatches.extend(one_microbatches)

    return chain(microbatches)


def get_batch(data_iterator, tuning):
    """Generate a batch."""

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    # return batch for GPT SFT
    if tuning:
        return data

    batch = {
        "tokens": data["tokens"],
        "labels": data["labels"],
        "loss_mask": data["loss_mask"],
        "attention_mask": data["attention_mask"],
        "position_ids": data["position_ids"],
    }
    if "weight" in data:
        batch.update({"weight": data["weight"]})
    if "avg_num_valid_tokens_in_ub" in data:
        batch.update({"avg_num_valid_tokens_in_ub": data["avg_num_valid_tokens_in_ub"]})
    return batch


class GPTSteerLMModel(GPTSFTModel):
    def loss_func(self, loss_mask, output_tensor):
        # loss is the nll loss for each token in the batch
        losses = output_tensor.float()
        # losses [batch_size, sequence_length]
        # calculate sum of log(p) for all tokens in a one batch item
        # this will be later used to calculate the baseline weights
        loss = torch.sum(losses * loss_mask, dim=-1)
        # if parallel_state.get_context_parallel_world_size() > 1:
        #     torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
        return loss

    def weight_loss_func(self, loss_mask, avg_num_valid_tokens_in_ub, output_tensor, weight):
        # loss is the nll loss for each token in the batch
        losses = output_tensor.float()
        loss_mask = loss_mask.float()
        # apply the per batch weight
        losses = losses * weight[:, None]
        # losses [batch_size, sequence_length]
        # loss per batch item, log(p) for all tokens in one batch item
        loss = torch.sum(losses * loss_mask, dim=-1)
        # sum log(p) in microbatch
        loss = loss.sum()
        # average lop(p) by the number of valid tokens in the microbatch and scale it by the number of microbatches
        loss = loss / avg_num_valid_tokens_in_ub[0]  # sequence level nll
        if parallel_state.get_context_parallel_world_size() > 1:
            torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
        return loss

    def get_forward_output_only_for_weight_func(self):
        def fwd_output_only_func(dataloader_iter, model):
            with torch.no_grad():
                batch = next(dataloader_iter)
                batch = {key: val.cuda(non_blocking=True) for key, val in batch.items()}

                forward_args = {
                    "input_ids": batch["tokens"],
                    "position_ids": batch["position_ids"],
                    "attention_mask": None if self.get_attention_mask_from_fusion else batch["attention_mask"],
                    "labels": batch["labels"],
                }

                output_tensor = model(**forward_args)

                def id_func(output_tensor):
                    nll = self.loss_func(batch["loss_mask"], output_tensor)
                    return output_tensor, {"loss": nll}

            return output_tensor, id_func

        return fwd_output_only_func

    def get_forward_output_and_loss_func(self, validation_step=False, tuning=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):

            # Get data batch
            batch = get_batch(dataloader_iter, tuning)

            # Transfer needed data to GPU
            required_keys = set()
            max_seqlen = batch["max_seqlen"].squeeze() if "max_seqlen" in batch else None
            cu_seqlens_argmin = batch["cu_seqlens_argmin"] if "cu_seqlens_argmin" in batch else None
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add("attention_mask")
                if "cu_seqlens" in batch:
                    required_keys.add("cu_seqlens")
                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(("tokens", "position_ids"))
                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(("labels", "loss_mask", "weight", "avg_num_valid_tokens_in_ub"))
            if self.get_attention_mask_from_fusion and "attention_mask" in required_keys:
                required_keys.remove("attention_mask")
            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            # slice batch along sequence dimension for context parallelism
            batch = self.get_batch_on_this_context_parallel_rank(batch)

            # Model forward pass
            forward_args = {
                "input_ids": batch["tokens"],
                "position_ids": batch["position_ids"],
                "attention_mask": None if self.get_attention_mask_from_fusion else batch["attention_mask"],
                "labels": batch["labels"],
                "loss_mask": batch["loss_mask"],
            }

            if not self.mcore_gpt:
                forward_args["checkpoint_activations_all_layers"] = checkpoint_activations_all_layers
                if not self.use_loss_mask:
                    forward_args.pop("loss_mask")
            else:
                # TODO: @eharper can we add this to mcore?
                forward_args.pop("loss_mask")

                if "cu_seqlens" in batch:  # packed sequence from GPTSFTPackedDataset
                    # these args are passed eventually into TEDotProductAttention.forward()
                    cu_seqlens = batch["cu_seqlens"].squeeze()  # remove batch size dimension (mbs=1)
                    # remove -1 "paddings" added in collate_fn
                    if cu_seqlens_argmin is not None:
                        cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
                    else:
                        cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

                    try:
                        from megatron.core.packed_seq_params import PackedSeqParams
                    except (ImportError, ModuleNotFoundError) as e:
                        mcore_version = packaging.version.Version(version("megatron-core"))
                        logging.error(
                            f"megatron-core v{mcore_version} does not support training with packed sequence. "
                            "Please use megatron-core >= 0.5.0, or set model.data.train_ds.packed_sequence=False"
                        )
                        raise e

                    forward_args["packed_seq_params"] = PackedSeqParams(
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_kv=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_kv=max_seqlen,
                        qkv_format="thd",
                    )

            output_tensor = model(**forward_args)

            def loss_func(output_tensor):
                # Loss for a micro-batch (ub)
                loss_for_ub = self.weight_loss_func(
                    batch["loss_mask"], batch["avg_num_valid_tokens_in_ub"], output_tensor, batch["weight"]
                )
                cp_size = parallel_state.get_context_parallel_world_size()
                if validation_step and not self.cfg.data.get("validation_drop_last", True):
                    num_valid_tokens_in_ub = batch["num_valid_tokens_in_ub"]
                    if loss_for_ub.isnan():
                        assert batch["loss_mask"].count_nonzero() == 0, "Got NaN loss with non-empty input"
                        loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
                    else:
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )
                    return loss_for_ub * cp_size, {"loss_sum_and_ub_size": loss_sum_and_ub_size_all_gpu}
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    return loss_for_ub * cp_size, {"avg": reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def get_loss_and_metrics(self, batch, forward_only):
        """Take a data_iter which is an interator over the microbatches
            and return loss as well as metrics
        """
        # first compute the baseline weights
        set_sync_funcs(self, True)
        distances = []

        # calculate the baseline weights for each group of generated reponses
        # set the batch size for log likelihood calculation
        response_size = batch["num_responses"][0].item()
        gbs, seq_length = batch["tokens"].shape
        mbs = int(self.cfg.data.steerlm2_weight_micro_batch_size)
        dp_size = 1
        configure_batch_sizes(mbs=mbs, gbs=gbs, dp=dp_size)

        group_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

        data_iter = get_iterator_k_split(group_batch, get_num_microbatches())

        fwd_bwd_function = get_forward_backward_func()

        # first get the forward nll loss for each of the microbatches
        fwd_loss_fn = self.get_forward_output_only_for_weight_func()
        losses_forward = fwd_bwd_function(
            forward_step_func=fwd_loss_fn,
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=True,
            micro_batch_size=get_micro_batch_size(),
            seq_length=seq_length,
        )
        if parallel_state.is_pipeline_last_stage():
            updated_weight = torch.concatenate([i["loss"] for i in losses_forward])

            batch["base_weight"] = torch.cuda.FloatTensor(gbs)
            batch["weight"] = torch.cuda.FloatTensor(gbs)
            batch["avg_num_valid_tokens_in_ub"] = torch.cuda.FloatTensor(gbs)

            # split updated weights into groups of size response_size
            for i in range(0, len(updated_weight), response_size):
                # the weights are negative log likelihood, to get log likelihood we need to negate the weights
                # compute base weight
                beg_idx = i
                end_idx = i + response_size
                base_weight = -updated_weight[beg_idx:end_idx] - batch["log(Q(y|a,x))"][beg_idx:end_idx].cuda()
                # for numerical stability, subtract the max value
                # calculate the probability vector
                base_weight = base_weight - base_weight.max()
                base_weight_p = torch.exp(base_weight)
                # noramliize the weight
                base_weight_p = base_weight_p / base_weight_p.sum()
                batch["base_weight"][beg_idx:end_idx] = base_weight_p
                ws_weight = batch["ws"][beg_idx:end_idx].cuda()
                batch["weight"][beg_idx:end_idx] = ws_weight - base_weight_p
                # group size is the number of responses generated
                loss_mbs = int(self.cfg.data.steerlm2_micro_batch_size)

                num_of_microbatches = gbs // loss_mbs
                # to compute the loss for each of the group batch,
                # loss L_i^{jb}  is the loss element for batch $b$, microbatch item $j$ and $i$ item in the microbatch
                # the loss we need is:  sum_{b=0}^{num_batches} sum_{j=0}^{num_of_microbatches} sum_{i=0}^{mbs} (L_i^{jb}) / num_batches ,
                # while the fwd-bwd function is used to compute the loss of average microbatch loss
                # i.e. L = sum_{b=0}^{num_batches} sum_{j=0}^{num_of_microbatches} [sum_{i=0}^{mbs} (L_i^{jb})] / (num_batches * num_of_microbatches)
                # to makie it equivalent to the loss we care, we need to scale loss by num_of_microbatches
                # i.e. L = sum_{b=0}^{num_batches} sum_{j=0}^{num_of_microbatches} [(num_of_microbatches) * sum_{i=0}^{mbs} (L_i^{jb})] / (num_batches * num_of_microbatches)
                # also if we want to average the loss weighted by the inverse of number of loss_mass=1 tokens in the batch
                batch["avg_num_valid_tokens_in_ub"][beg_idx:end_idx] = (
                    batch["loss_mask"][beg_idx:end_idx].sum(axis=-1).float().mean().repeat(response_size)
                    / num_of_microbatches
                )
                # compute the kl divergence between group['ws'] and group['base_weight']
                # for numerical stability, add a small value to the base_weight_p
                base_weight_p = base_weight_p + 1e-8
                ws_weight = ws_weight + 1e-8
                distance = torch.sum(ws_weight * torch.log(ws_weight / base_weight_p))
                distances.append(distance)
            # get average distance
            distance = torch.stack(distances).mean()
            distance = average_losses_across_data_parallel_group([distance])

        # compute the gbs which are all the responses generated
        # dp_size = int(parallel_state.get_data_parallel_world_size())
        # gbs = len(batch) * dp_size * sizes[0]
        # mbs = int(self.cfg.data.steerlm2_micro_batch_size)
        # configure_batch_sizes(mbs=mbs, gbs=gbs, dp=dp_size)
        # restore the batch sizes for training
        set_sync_funcs(self, forward_only)

        dp_size = int(parallel_state.get_data_parallel_world_size())
        gbs = dp_size * gbs
        mbs = int(self.cfg.data.steerlm2_micro_batch_size)
        configure_batch_sizes(mbs=mbs, gbs=gbs, dp=dp_size)

        # modify the iterator_k_split to iterate over all the data
        # data_iter = get_grouped_iterator_k_split(group, sub_num_microbatches)
        data_iter = get_iterator_k_split(batch, get_num_microbatches())

        fwd_loss_fn = self.get_forward_output_and_loss_func(forward_only)

        losses_reduced = fwd_bwd_function(
            forward_step_func=fwd_loss_fn,
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            micro_batch_size=get_micro_batch_size(),
            seq_length=seq_length,
        )
        torch.cuda.synchronize()

        # only the last stages of the pipeline return losses
        if parallel_state.is_pipeline_last_stage():
            loss_mean = torch.concat([loss["avg"] for loss in losses_reduced]).mean()
        else:
            loss_mean = torch.tensor(0.0).cuda()
            distance = torch.tensor(0.0).cuda()
        # Logging
        torch.distributed.broadcast(loss_mean, get_last_rank())
        torch.distributed.broadcast(distance, get_last_rank())
        loss_value = loss_mean.detach().item()
        metrics = {"loss": loss_value, "distance": distance.detach().item()}
        return loss_value, metrics
