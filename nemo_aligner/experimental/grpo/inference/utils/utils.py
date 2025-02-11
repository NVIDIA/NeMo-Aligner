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
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from safetensors.torch import save_file
import os

def append_and_repad_list(list_of_items: List, item_to_append, pad_id):
    """
    Appends an item to a list and increases its size by 1, while ensuring padding consistency.

    Args:
        list_of_items (list): The original input list of items.
        item_to_append: The item to append to the list.
        pad_id: The padding ID used to fill the list to the desired size.

    Returns:
        list: A new list with the appended item, increasing the size of the input list by one.
    """
    # Remove all elements equal to the padding ID from the list
    items = [item for item in list_of_items if item != pad_id]
    
    # Append the new item to the filtered list
    items.append(item_to_append)
    
    # Ensure the result is "list_of_items size + 1", using pad_id for extra padding if necessary
    if len(items) < len(list_of_items) + 1:
        items += [pad_id] * (len(list_of_items) + 1 - len(items))

    return items

def save_chunk(chunk, filename):
    save_file(chunk, filename)

def parallel_save_cpu_state_dict(state_dict, path, num_chunks=4):
    # Split the state dict into N chunks (here we choose 4)
    keys = list(state_dict.keys())
    chunk_size = (len(keys) + num_chunks - 1) // num_chunks

    chunks = []
    for i in range(num_chunks):
        chunk_keys = keys[i * chunk_size: (i + 1) * chunk_size]
        chunk = {k: state_dict[k] for k in chunk_keys}
        chunks.append(chunk)

    filenames = [os.path.join(path, f"model_chunk_{i}.safetensors") for i in range(num_chunks)]

    # Save each chunk concurrently
    with ThreadPoolExecutor(max_workers=num_chunks) as executor:
        futures = [executor.submit(save_chunk, chunks[i], filenames[i])
                   for i in range(num_chunks)]
        for future in as_completed(futures):
            future.result()  # Ensure any exceptions are raised

    print("State dict saved in chunks:", filenames)

    for filename in filenames:
        os.chmod(filename, 0o666)
