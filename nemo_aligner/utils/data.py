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


## copied directly from https://github.com/NVIDIA/NeMo/blob/ca4e4f0d7ce9f11be7bb79d8dba42ee53b7991ad/nemo/collections/nlp/data/language_modeling/megatron/base_dataset_utils.py#L18

import math

## from https://github.com/NVIDIA/NeMo/blob/cc365b6c1fd4d93994d2fb79ac26c44a5e717776/nemo/collections/nlp/modules/common/megatron/utils.py#L206
## different from mcore's _get_ltor_masks_and_position_ids
def get_ltor_masks_and_position_ids(
    data, eod_token, reset_position_ids, reset_attention_mask, eod_mask_loss, compute_attention_mask=True
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    attention_mask = None
    if compute_attention_mask:
        attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length
        )

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indicies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    if compute_attention_mask:
        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def get_train_valid_test_split_(splits_string, size):
    """ Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]
    if len(splits) != 3:
        raise ValueError(f"Invalid splits string: {splits_string}. Expected 3 comma separated values.")
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index

## copied from https://github.com/NVIDIA/NeMo/blob/ca4e4f0d7ce9f11be7bb79d8dba42ee53b7991ad/nemo/collections/nlp/data/language_modeling/megatron/indexed_dataset.py#L380
## TODO: is there a megatron class we can use instead?
import numpy as np
import struct
import torch

from functools import lru_cache
from itertools import accumulate

def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass

def index_file_path(prefix_path):
    return prefix_path + '.idx'

def data_file_path(prefix_path):
    return prefix_path + '.bin'

class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, 'wb')

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack('<Q', 1))
                    self._file.write(struct.pack('<B', code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack('<Q', len(sizes)))
                    self._file.write(struct.pack('<Q', len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order='C'))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order='C'))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                logging.info("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            logging.info("    reading sizes...")
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            logging.info("    reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes
            )
            logging.info("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False, delay_data_mmap=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None
        self._delay_data_mmap = delay_data_mmap
        self._skip_warmup = skip_warmup

        self._do_init(path, skip_warmup, delay_data_mmap)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup=True, delay_data_mmap=False):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not delay_data_mmap:
            self._create_data_mmap(skip_warmup)
        else:
            logging.info("    skip creating data numpy buffer of mmap...")
            self._bin_buffer_mmap = None
            self._bin_buffer = None

    def _create_data_mmap(self, skip_warmup):
        if not skip_warmup:
            logging.info("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        logging.info("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        logging.info("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)
        return np_array

    def create_data_mmap(self):
        self._create_data_mmap(self._skip_warmup)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))

## copied from https://github.com/NVIDIA/NeMo/blob/ca4e4f0d7ce9f11be7bb79d8dba42ee53b7991ad/nemo/collections/nlp/data/language_modeling/megatron/indexed_dataset.py#L92
## simplified for aligner
def make_dataset(path, impl, skip_warmup=False, impl_kwargs={}, delay_data_mmap=False):
    if impl == 'mmap' and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path, skip_warmup, delay_data_mmap)
    raise ValueError(f"Unknown dataset implementation: {impl}")

## copied from https://github.com/NVIDIA/NeMo/blob/ca4e4f0d7ce9f11be7bb79d8dba42ee53b7991ad/nemo/collections/nlp/data/language_modeling/megatron/gpt_dataset.py#L279
def get_indexed_dataset_(data_prefix, data_impl, skip_warmup, delay_data_mmap=False):
    """Build indexed dataset."""
    logging.info(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_dataset(data_prefix, data_impl, skip_warmup, delay_data_mmap=delay_data_mmap)
    logging.info(' > finished creating indexed dataset in {:4f} ' 'seconds'.format(time.time() - start_time))
    logging.info('    number of documents: {}'.format(indexed_dataset.sizes.shape[0]))

    return indexed_dataset


## copied from https://github.com/NVIDIA/NeMo/blob/ad807ae56821c638923f20a251694c1fdac6272f/nemo/collections/nlp/data/language_modeling/megatron/megatron_batch_samplers.py#L176
import wrapt
from nemo.utils import logging

from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import BaseMegatronBatchSampler ## TODO: port to nemo2. Nemo2 also has a dep on this

@wrapt.decorator
def experimental(wrapped, instance, args, kwargs):
    logging.warning(f"`{wrapped}` is experimental and not ready for production yet. Use at your own risk.")
    return wrapped(*args, **kwargs)

@experimental
class MegatronPretrainingRandomBatchSampler(BaseMegatronBatchSampler):

    # NOTE (mkozuki): [[Argument of `dataset` and `data_sharding`]]
    # From the commit below, it seems like `dataset` argument and `data_sharding` argument
    # are necessary for ViT training. However, to keep this simple,
    # I omit those two arguments.
    # commit: https://github.com/NVIDIA/Megatron-LM/commit/7a77abd9b6267dc0020a60b424b4748fc22790bb
    #
    # NOTE (degert): I have re-written this class somewhat to give the length correctly when consumed_samples
    # are larger than total_samples, which happens with epochs > 1 training when using this Sampler
    # I have also added an explicit seed which allows us to remove Dataset-side shuffling in Nemo-Aligner
    #
    # This class does not currently work with pad_samples_to_global_batch_size=True
    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        global_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool,
        pad_samples_to_global_batch_size: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(
            total_samples=total_samples,
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=drop_last,
            global_batch_size=global_batch_size,
            pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
        )
        assert (
            not pad_samples_to_global_batch_size
        ), "`MegatronPretrainingRandomBatchSampler` does not support sample padding"
        if (not drop_last) and self.micro_batch_times_data_parallel_size > 1:
            raise RuntimeError(
                "`MegatronPretrainingRandomBatchSampler` does not support drop_last=False when micro_batch_size * data_parallel_size > 1. \
                  please reduce your MBS and data parallelism to 1 if you want to use drop_last=False, or switch to drop_last=True to avoid this error"
            )
        self.last_batch_size = self.total_samples % self._global_batch_size
        self.seed = seed

    def __len__(self) -> int:
        """Length of Random Batch Sampler.

        ..note::
            When `rampup_batch_size` is enabled, the return value can be not exactly precise.

        """
        active_total_samples = self.total_samples - (self.last_batch_size if self.drop_last else 0)
        num_available_samples = active_total_samples - self.consumed_samples % active_total_samples
        if self.drop_last:
            return num_available_samples // self.global_batch_size
        else:
            return (num_available_samples + self.global_batch_size - 1) // self.global_batch_size

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self._global_batch_size_on_this_data_parallel_rank:
                self.consumed_samples += self._global_batch_size
                yield batch
                batch = []
        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch


## copied from https://github.com/NVIDIA/NeMo/blob/ad807ae56821c638923f20a251694c1fdac6272f/nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_chat_dataset.py#L39
## TODO: see whether we can remove this once GPTSFTChatDataset has been ported to nemo2
SYSTEM_TOKEN = "System"
TYPE_INSTRUCTION = {
    'TEXT_TO_VALUE': "",
    'VALUE_TO_TEXT': '',
}

def _add_speaker_and_signal(header, source, mask_role, gtype, special_tokens):
    TURN_TOKEN = special_tokens['turn_start']
    END_SIGNAL = special_tokens['end_of_turn']
    LABEL_START = special_tokens['label_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = ""
    conversation = header
    for i, sentence in enumerate(source):
        sentence_from = sentence["from"]
        role_token = TURN_TOKEN
        if gtype is None:
            sentence["value"] = (
                BEGIN_SIGNAL + role_token + sentence_from + END_NAME_SIGNAL + sentence["value"] + END_SIGNAL
            )
        elif gtype == "VALUE_TO_TEXT":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + (
                    response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                    if 'label' in sentence
                    else ''
                )
                + sentence["value"]
                + END_SIGNAL
            )
        elif gtype == "TEXT_TO_VALUE":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + sentence["value"]
                + END_SIGNAL
                + (
                    response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                    if 'label' in sentence
                    else ''
                )
            )
        else:
            raise ValueError(
                f"source type {gtype} not supported, only 'VALUE_TO_TEXT' and 'TEXT_TO_VALUE' are supported"
            )
        conversation += sentence["value"]
        # if the last turn is not masked, add next token start token to the end, which will be included for loss calculation
        if sentence_from not in mask_role and i == len(source) - 1:
            conversation += TURN_TOKEN
    return conversation

def _get_header_conversation_type_mask_role(source, special_tokens):
    END_SIGNAL = special_tokens['end_of_turn']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    data_type = None
    if 'type' in source:
        data_type = source['type']
        if data_type is not None:
            assert data_type in TYPE_INSTRUCTION, f"source type {data_type} not supported"
    # add end signal and concatenate together
    conversation = source['system']
    if data_type is not None:
        if TYPE_INSTRUCTION[data_type] != '':
            conversation = conversation + '\n' + TYPE_INSTRUCTION[data_type]
    mask_role = source.get('mask', 'User')
    header = f"{special_tokens['system_turn_start']}{SYSTEM_TOKEN}{END_NAME_SIGNAL}{conversation}{END_SIGNAL}"
    conversation = _add_speaker_and_signal(header, source['conversations'], mask_role, data_type, special_tokens)
    return header, conversation, data_type, mask_role

def get_prompt_template_example(special_tokens):
    source = {
        'system': '{system message}',
        'conversations': [
            {'from': 'User', 'value': '{turn 1 user message}', 'label': None},
            {'from': 'Assistant', 'value': '{turn 1 assistant message}', 'label': '{turn 1 assistant label}'},
            {'from': 'User', 'value': '{turn 2 user message}', 'label': None},
            {'from': 'Assistant', 'value': '{turn 2 assistant message}', 'label': '{turn 2 assistant label}'},
        ],
        "mask": "User",
        "type": "VALUE_TO_TEXT",
    }
    _, conversation, _, _ = _get_header_conversation_type_mask_role(source, special_tokens)
    return conversation