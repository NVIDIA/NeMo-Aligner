# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
from typing import Union, Dict, List, Optional, Iterator
from nemo.utils import logging
from nemo.collections.nlp.modules.common.megatron.utils import split_list
import torch

class TensorList:
    """
    A data structure that hold a list of tensors. 
    Provides the batching capability with nested tensors, which PyTorch is lacking as of 10/01/2024.
    """
    def __init__(self, list_of_tensors):
        self.tensors = list_of_tensors

    def __getitem__(self, idx):
        # Allows indexing to get a specific tensor
        return self.tensors[idx]

    def __len__(self):
        # Return the number of tensors
        return len(self.tensors)

    def to(self, *args, **kwargs):
        # Apply the 'to' method to each tensor in the list
        return TensorList([nt.to(*args, **kwargs) for nt in self.tensors])

    def cuda(self, non_blocking=False):
        """
        Convenience method to move TensorList to CUDA.
        """
        return self.to('cuda', non_blocking=non_blocking)

    @classmethod
    def cat(cls, lists):
        """
        Concatenates multiple TensorList instances.

        Args:
            lists (list of TensorList): List of TensorList instances to concatenate.

        Returns:
            TensorList: A new TensorList containing tensors from all lists.

        Raises:
            TypeError: If any element in 'lists' is not an instance of TensorList.
        """
        if not all(isinstance(lst, TensorList) for lst in lists):
            raise TypeError("All elements to concatenate must be TensorList instances")
        
        concatenated = []
        for lst in lists:
            concatenated.extend(lst.tensors)
        
        return cls(concatenated)
    
    def __repr__(self):
        # Custom representation for printing
        return f"TensorList({self.tensors})"
    
def get_iterator_k_split(
    batch: Union[Dict, List[torch.Tensor], TensorList], 
    num_microbatches: int, 
    enforce_divisible_batch: Optional[bool] = True
) -> Iterator:
    """
    Split a batch into k microbatches, where the batch size is divisible by k. Batch could be
    a dictionary of tensors or a list of tensors. A dictionary batch could also have items of List type,
    as long as the length of that list is the same as the batch size.
    
    Now supports TensorList by handling nested tensors correctly.
    """
    if isinstance(batch, dict):
        # Filter out unsupported items
        discard_items = [k for k, v in batch.items() if not isinstance(v, (torch.Tensor, list, TensorList))]
        if len(discard_items) > 0:
            logging.warning(
                f"Only support splitting torch.Tensor, List[torch.Tensor], and TensorList. "
                f"Discarding the following keys from the batch: {discard_items}",
            )

        batch = {k: v for k, v in batch.items() if isinstance(v, (torch.Tensor, list, TensorList))}
        
        tensor_items = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        list_items = {k: v for k, v in batch.items() if isinstance(v, list)}
        tensor_list_items = {k: v for k, v in batch.items() if isinstance(v, TensorList)}

        # Split tensor items
        items = list(tensor_items.items())
        if enforce_divisible_batch:
            assert items[0][1].shape[0] % num_microbatches == 0, "Issue with batch size configuration!"
        split_batch = [torch.tensor_split(item[1], num_microbatches, dim=0) for item in items]

        # Handle the case where the batch size from dynamic bucketing is not divisible
        if items[0][1].shape[0] % num_microbatches != 0:
            chunk_size = split_batch[0][-1].shape[0]
            split_batch = [[j[:chunk_size] for j in i] for i in split_batch]

        # Split TensorList items
        tensor_list_items = list(tensor_list_items.items())
        split_tensor_list_batch = [
            [v.tensors[i] for i in range(num_microbatches)] 
            for k, v in tensor_list_items
        ]

        if len(list_items) == 0 and len(tensor_list_items) == 0:
            # Only have tensor items
            microbatches = [
                {items[i][0]: split_batch[i][j] for i in range(len(items))} for j in range(num_microbatches)
            ]
        else:
            # Split list items
            list_items = list(list_items.items())
            split_list_batch = [
                split_list(item[1], num_microbatches, enforce_divisible_batch=enforce_divisible_batch)
                for item in list_items
            ]
       
            # Merge tensor, list, and tensor_list (possibly nested tensor) items
            all_keys = [item[0] for item in items] + [item[0] for item in list_items] + [item[0] for item in tensor_list_items]
            all_split_batch = split_batch + split_list_batch + split_tensor_list_batch
            microbatches = [
                {all_keys[i]: all_split_batch[i][j] for i in range(len(all_keys))} for j in range(num_microbatches)
            ]

    elif isinstance(batch, list):
        # Split a list of torch tensors or TensorList
        assert batch[0].shape[0] % num_microbatches == 0, "Issue with batch size configuration!"
        split_batch = [
            torch.tensor_split(item, num_microbatches, dim=0) if isinstance(item, torch.Tensor) else item
            for item in batch
        ]
        microbatches = [
            [elem[i] if elem is not None else elem for elem in split_batch] for i in range(num_microbatches)
        ]
    return itertools.chain(microbatches)