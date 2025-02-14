#!/usr/bin/env python
import ctypes
import json
import sys
import threading
import time

import numpy as np
import requests
import torch
from multiprocessing import shared_memory

# --- Shared Memory & Tensor Dictionary Implementation ---

# Mapping from torch dtypes to numpy dtypes used for shared memory.
# Note: torch.bfloat16 is not a native numpy dtype, so we store its raw bits in np.uint16.
TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32:   np.int32,
    torch.int64:   np.int64,
    torch.uint8:   np.uint8,
    torch.int8:    np.int8,
    torch.float16: np.float16,
    torch.bfloat16: np.uint16,  # bfloat16 stored as 16-bit unsigned ints.
}

STR_TO_TORCH_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "float16": torch.float16,
}

class SharedTensorEntry:
    def __init__(self, shm_name, shape, torch_dtype):
        self.shm_name = shm_name
        self.shape = shape
        self.torch_dtype = torch_dtype


def mlock_buffer(buf):
    """
    Pins the memory for the given buffer using OS calls (mlock on Linux, VirtualLock on Windows).
    This ensures the memory remains resident (page-locked). Note: mlock is per-process.
    """
    return
    address = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    size = len(buf)
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL("libc.so.6")
        res = libc.mlock(ctypes.c_void_p(address), ctypes.c_size_t(size))
        if res != 0:
            raise OSError("mlock failed")
    else:
        print("WARNING:Pinned memory not implemented for this platform")
        return


class SharedCPUMemoryTensorDict:
    """
    A dictionary-like object that maps keys to shared, pinned CPU tensors.
    
    Process 0 (the master) can add or update tensors.
    The shared memory block is either reused (if the shape/dtype match) or reallocated.
    Metadata is maintained for sharing with external processes.
    """
    def __init__(self, metadata=None, communicable_metadata=None):
        """
        If metadata is provided it should be a dict mapping keys to SharedTensorEntry.
        If communicable_metadata is provided it should be a dict mapping keys to SharedTensorEntry fields in string form.
        Provide either one, or neither
        This allows a process to attach to an already published shared tensor dictionary.
        """
        # check that at most one of metadata or communicable_metadata is provided
        assert (not metadata and not communicable_metadata) or ((metadata is not None) ^ (communicable_metadata is not None)), "Provide at most one of metadata or communicable_metadata"
        if metadata is not None:
            self.metadata = metadata
        elif communicable_metadata is not None:
            self.metadata = {key: SharedTensorEntry(entry["shm_name"], tuple(entry["shape"]), STR_TO_TORCH_DTYPE[entry["dtype"]]) for key, entry in communicable_metadata.items()}
        else:
            self.metadata = {}
        self._shm_cache = {}  # Cache of open shared_memory.SharedMemory objects.
        self._np_cache = {}  # Cache of numpy arrays for gc

    def __setitem__(self, key, tensor: torch.Tensor):
        #if tensor.device.type != "cpu":
        #    raise ValueError("Only CPU tensors are supported")
        shape = tuple(tensor.shape)
        torch_dtype = tensor.dtype
        if torch_dtype not in TORCH_TO_NUMPY:
            raise ValueError(f"Unsupported tensor dtype: {torch_dtype}")
        np_dtype = TORCH_TO_NUMPY[torch_dtype]
        new_nbytes = int(np.prod(shape) * np.dtype(np_dtype).itemsize)

        # If the key exists and the new tensor matches in shape and dtype, update the existing buffer.
        if key in self.metadata:
            existing_entry = self.metadata[key]
            if shape == existing_entry.shape and torch_dtype == existing_entry.torch_dtype:
                shm = self._shm_cache.get(key)
                if shm is None:
                    shm = shared_memory.SharedMemory(name=existing_entry.shm_name)
                    self._shm_cache[key] = shm
                if shm.size < new_nbytes:
                    raise ValueError("Existing shared memory block is smaller than new tensor size")
                np_array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)
                torch_numpy_view = torch.from_numpy(np_array)
                if torch_dtype == torch.bfloat16:
                    torch_numpy_view.copy_(tensor.view(torch.uint16))
                    # np.copyto(np_array, tensor.view(torch.uint16).numpy())
                else:
                    torch_numpy_view.copy_(tensor)
                    # np.copyto(np_array, tensor.numpy())
                # Pin the memory in this process.
                #print(f"Found existing shm, setting {key} {shm.name}")
                try:
                    mlock_buffer(shm.buf)
                except Exception as e:
                    print("Warning: mlock failed:", e)
                return
            else:
                # Shape or dtype changed – close and unlink the old shared memory.
                old_shm = self._shm_cache.get(key)
                if old_shm is not None:
                    try:
                        old_shm.close()
                        old_shm.unlink()
                    except FileNotFoundError:
                        pass
                    del self._shm_cache[key]

        # Allocate a new shared memory block.
        shm = shared_memory.SharedMemory(create=True, size=new_nbytes)
        #print(f"shm address: {ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))}", flush=True)
        np_array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)
        self._np_cache[key] = np_array
        torch_numpy_view = torch.from_numpy(np_array)
        if torch_dtype == torch.bfloat16:
            # np.copyto(np_array, tensor.view(torch.uint16).numpy())
            torch_numpy_view.copy_(tensor.view(torch.uint16))
        else:
            # np.copyto(np_array, tensor.numpy())
            torch_numpy_view.copy_(tensor)
        self._shm_cache[key] = shm
        self.metadata[key] = SharedTensorEntry(shm.name, shape, torch_dtype)
        # Pin the memory.
        try:
            mlock_buffer(shm.buf)
        except Exception as e:
            print("Warning: mlock failed:", e)

    def __getitem__(self, key):
        """
        Returns a torch tensor that is a zero‑copy view on the shared, pinned memory.
        """
        if key not in self.metadata:
            raise KeyError(f"Key {key} not found in SharedTensorDict")
        entry = self.metadata[key]
        print(f"Getting {key} {entry.shm_name} {entry.shape} {entry.torch_dtype}", flush=True)
        if key in self._shm_cache:
            shm = self._shm_cache[key]
        else:
            #print(f"Binding new shm for {key} {entry.shm_name}", flush=True)
            shm = shared_memory.SharedMemory(name=entry.shm_name)
            #print(f"shm address: {ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))}", flush=True)
            self._shm_cache[key] = shm

        np_dtype = TORCH_TO_NUMPY[entry.torch_dtype]
        np_array = np.ndarray(entry.shape, dtype=np_dtype, buffer=shm.buf)
        self._np_cache[key] = np_array
        # Pin the memory in this process.
        try:
            mlock_buffer(shm.buf)
        except Exception as e:
            print("Warning: mlock failed:", e)

        if entry.torch_dtype == torch.bfloat16:
            tensor = torch.from_numpy(np_array).view(torch.bfloat16)
        else:
            tensor = torch.from_numpy(np_array)
        return tensor

    def as_dict(self):
        return {key: self[key] for key in self.metadata.keys()}

    def keys(self):
        return self.metadata.keys()

    def close(self):
        """Close all open shared memory handles."""
        for shm in self._shm_cache.values():
            shm.close()
        self._shm_cache.clear()

    def unlink(self):
        """
        Unlink (destroy) the shared memory blocks.
        Only call this from the process that created the shared memory.
        """
        for shm in self._shm_cache.values():
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        self._shm_cache.clear()

    def get_metadata_dict(self):
        return {key: {
            "shm_name": entry.shm_name,
            "shape": entry.shape,
            "dtype": str(entry.torch_dtype).split(".")[-1]
        } for key, entry in self.metadata.items()}

