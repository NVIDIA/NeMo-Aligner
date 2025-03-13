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

# Dictionary to hold registered backends
_BACKEND_REGISTRY = {}

def register_backend(name):
    """Decorator to register a new backend."""
    def decorator(backend_cls):
        _BACKEND_REGISTRY[name] = backend_cls
        return backend_cls
    return decorator

def get_backend(name, *args, **kwargs):
    """
    Factory method to initialize and return the requested backend.
    Args:
        name (str): Name of the registered backend.
        *args, **kwargs: Arguments to pass to the backend constructor.
    """
    if name not in _BACKEND_REGISTRY:
        raise ValueError(f"Backend '{name}' is not registered.")
    return _BACKEND_REGISTRY[name](*args, **kwargs)

def list_available_backends():
    """
    Returns a list of all registered backends.
    """
    return list(_BACKEND_REGISTRY.keys())
