from __future__ import annotations

'''Importing the type of generator to use'''

from typing import Any, Union

from lactchain.classes.base_generator import LLMGenerator
from lactchain.generators.vllm_backend import VLLMGenerator, VLLMGeneratorConfig

GeneratorConfigs=Union[
    VLLMGeneratorConfig
]

STRATEGIES={
    
    'vllm': (VLLMGeneratorConfig, VLLMGenerator)
    
}

# pass in generatorconfig via .model_dump()
def generator_factory(pretrained_model_name_or_path: str, config: GeneratorConfigs) -> LLMGenerator:
    '''Generator factory that takes in config and selects 
    generator and config with the kwargs added in
    
    Input: 
    =====
    kwargs: dict[str, Any]
        An instance of a GeneratorConfig with 'name' as the generators
    '''
    
    kwargs = config.model_dump()
    kwargs['pretrained_model_name_or_path'] = pretrained_model_name_or_path
    
    name = kwargs.get('name', '')
    strategy = STRATEGIES.get(name)  # type: ignore[arg-type]
    if not strategy:
        raise ValueError(
            f'Unknown generator name: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )

    # Get the config and classes
    config_cls, cls = strategy

    return cls(config_cls(**kwargs))