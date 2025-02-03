
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
