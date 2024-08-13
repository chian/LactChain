"""Generator module."""

from __future__ import annotations

from typing import Any
from typing import Union

from lactchain.classes.base_generator import LLMGenerator
from lactchain.registry.registry import registry
from lactchain.generators.vllm_backend import VLLMGenerator, VLLMGeneratorConfig

LLMGeneratorConfigs=Union[
    VLLMGeneratorConfig
]

GENERATORS={
    'vllm': (VLLMGeneratorConfig, VLLMGenerator)
}

def _factory_fn(**kwargs: dict[str, Any]) -> LLMGenerator:
    name = kwargs.get('name', '')
    generator = GENERATORS.get(name)  # type: ignore[arg-type]
    if not generator:
        raise ValueError(
            f'Unknown generator name: {name}.'
            f' Available: {set(GENERATORS.keys())}',
        )

    # Get the config and classes
    config_cls, cls = generator

    return cls(config_cls(**kwargs))


def get_generator(
    kwargs: dict[str, Any],
    register: bool = False,
) -> LLMGenerator:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - vllm
    - langchain
    - huggingface

    Parameters
    ----------
    kwargs : dict[str, Any]
        The configuration. Contains a `name` argument
        to specify the strategy to use.
    register : bool, optional
        Register the instance for warmstart. Caches the
        instance based on the kwargs, by default False.

    Returns
    -------
    LLMGenerator
        The instance.

    Raises
    ------
    ValueError
        If the `name` is unknown.
    """
    # Create and register the instance
    if register:
        registry.register(_factory_fn)
        return registry.get(_factory_fn, **kwargs)

    return _factory_fn(**kwargs)