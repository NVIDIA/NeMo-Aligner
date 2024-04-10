# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


def decode_bytes_ndarray(str_ndarray: np.ndarray) -> np.ndarray:
    str_ndarray = str_ndarray.astype("bytes")
    str_ndarray = np.char.decode(str_ndarray, encoding="utf-8")
    return str_ndarray.squeeze(axis=-1)


def lock_method(lock_name):
    """
    Decorator to use in a class to ensure only one method is executing at a time.

    For instance:

        class MyClass:

            def __init__(self):
                self.my_lock = threading.Lock()

            @lock_method("self.my_lock")
            def method1(self):
                return ...

            @lock_method("self.my_lock")
            def method2(self):
                return ...
    """
    # We enforce the usage of the "self." prefix to make it explicit where the lock comes from.
    prefix = "self."
    assert lock_name.startswith(prefix), f"`lock_name` ({lock_name}) must start with '{prefix}'"
    lock_name = lock_name[len(prefix) :]

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            with getattr(self, lock_name):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


def pad_input(value: Optional[np.ndarray], size: int, pad_value: int = 0):
    """pad the input to a multiple of `size` and return it as a list"""
    extra = 0
    if value is not None:
        if value.dtype == bytes:
            value = decode_bytes_ndarray(value)
        if value.shape[0] % size != 0:
            extra = size - (value.shape[0] % size)

            pad_width = [(0, extra)] + [(0, 0)] * (value.ndim - 1)
            value = np.pad(value, pad_width=pad_width, mode="constant", constant_values=pad_value)
        value = value.tolist()
    return value, extra


class FutureResult(ABC):
    """Generic class for trainers to wait if an object is an instance of this class
    """

    @abstractmethod
    def result(self):
        """called by the trainer to get the result must be broadcasted to all ranks
        """
