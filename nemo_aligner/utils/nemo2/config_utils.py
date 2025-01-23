from dataclasses import dataclass, fields, make_dataclass
from typing import Optional

from nemo.collections.llm.gpt.model.base import GPTConfig

def create_optional_config(base_class):
    field_definitions  = [
        (field.name, Optional[field.type], None)
        for field in fields(base_class)
    ]

    return make_dataclass(f"{base_class.__name__}Overrides", field_definitions)

## gives a class with same attributes as GPTConfig, but
## defaults value of all attributes is None
GPTConfigOverrides = create_optional_config(GPTConfig)

def maybe_override(base_config, overwrite_config):
    for field in fields(base_config.__class__):
        new_attr = getattr(overwrite_config, field.name)
        if new_attr is not None:
            setattr(base_config, field.name, new_attr)
    return base_config