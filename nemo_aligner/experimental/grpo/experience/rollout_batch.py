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
from collections import UserDict
from typing import List, Optional, Dict
from typing_extensions import Self

import torch

from nemo_aligner.experimental.grpo.utils import parallel_state
from nemo_aligner.utils.distributed import rebalance_nd_tensor, gather_jagged_object_lists

class GPTRolloutBatch(UserDict):
    @classmethod
    def from_rollout_batches(
        cls: Self, rollout_batches: List[Dict], eos_id: int, rollout_batch_seq_length: Optional[int]
    ) -> Self:
        """
        Given a list of rollout batches, stack the tensors/lists within and put them in a single dictionary
        """
        stacked_dict = cls()

        for k in sorted(rollout_batches[0]):

            list_of_tensors = [item[k] for item in rollout_batches]

            if isinstance(list_of_tensors[0], list):
                tensor = [item for sublist in list_of_tensors for item in sublist]
            elif all(x.ndim == 1 for x in list_of_tensors):
                tensor = torch.cat(list_of_tensors)
            else:
                print(k, flush=True)
                pad_value = eos_id if k == "response_tokens" else 0

                list_of_tensors = [row.flatten() for tensor in list_of_tensors for row in tensor]
                # TODO: can we avoid padding locally then padding globally?
                tensor = torch.nn.utils.rnn.pad_sequence(list_of_tensors, batch_first=True, padding_value=pad_value)

                # find the max sequence length globally
                max_seqlen = torch.tensor([tensor.size(-1)], dtype=torch.long, device=torch.cuda.current_device())
                torch.distributed.all_reduce(max_seqlen, op=torch.distributed.ReduceOp.MAX)

                if rollout_batch_seq_length is None or max_seqlen >= rollout_batch_seq_length:
                    pad_seq_len = max_seqlen.item()
                else:
                    # response tokens must be B x S because computing log probs requires us to offset by 1
                    pad_seq_len = rollout_batch_seq_length if k == "response_tokens" else rollout_batch_seq_length - 1

                tensor = torch.nn.functional.pad(tensor, (0, pad_seq_len - tensor.size(-1)), value=pad_value)

            stacked_dict[k] = tensor

        return stacked_dict

    def gather_and_balance_globally(self):
        """
        Gathers batches with jagged leading dimensions across the DP ranks. If using reshard, it will treat PP as DP ranks.
        Works with data that is either tensors or string lists.
        """
        global_rollout_batch = type(self)()

        for k, value in self.data.items():
            if isinstance(value, torch.Tensor):
                # With resharding for inference enabled, PP groups in training turn into DP groups for inference. 
                # So, we need to balance them first and then balance by all the original (training) DP groups
                # We call get_(training)_pipeline_model_parallel_group() here because that refers to the pp groups
                # as they were in training. In the inference(current) context, get_pipeline_model_parallel_group would be 1.
                value = rebalance_nd_tensor(value, group=parallel_state.get_data_parallel_group())
                global_rollout_batch[k] = value
            elif isinstance(value, list):
                # same infernence reshard logic described above, but now using object gathers.
                value = gather_jagged_object_lists(value, parallel_state.get_data_parallel_group())
                global_rollout_batch[k] = value
            else:
                raise NotImplementedError(
                    (
                        f"Attempted to gather_and_balance_globally for unsupported type {type(value)} with key {k}."
                        "Please provide either a tensor or a list of picklable objects."
                    )
                )

        return global_rollout_batch

    def chunk(self, rank, split_size):
        """
        Chunks a global batch into splits of size split_size and returns the 'rank'th split
        batch=[A A A B B B D D E], rank=2, split_size=3 -> [D D E] 

        Requires all leading dimensions of tensors and lengths of lists to be the same over the batch
        and the split_size must divide batch size.
        """
        chunked_rollout_batch = type(self)()

        batch_set = set()
        for val in self.data.values():
            if isinstance(val, torch.Tensor):
                batch_set.add(val.size(0))
            else:
                batch_set.add(len(val))

        assert len(batch_set) == 1, "batch sizes are not the same across the rollout batch"
        B = batch_set.pop()
        assert B % split_size == 0, f"batch size ({B}) is not a multiple of split_size ({split_size})"
        assert split_size > rank, \
            f"index OOB: not enough splits for this rank. rollout_batch_size: {B}, split_size ({split_size}), rank_idx ({rank})"

        indices = torch.arange(B).tensor_split(split_size)[rank]

        for k in self.data:
            if torch.is_tensor(self.data[k]):
                chunked_rollout_batch[k] = self.data[k][indices].clone()
            else:
                chunked_rollout_batch[k] = [self.data[k][i] for i in indices]

        return chunked_rollout_batch

    def get_dict(self):
        return self.data
