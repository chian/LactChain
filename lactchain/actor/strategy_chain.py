from __future__ import annotations

'''Chain for action inference'''

from textwrap import dedent
from typing import Any, List, Dict, Optional, Literal, Tuple, TypeVar, Callable
from pydantic import BaseModel, Field
import torch
import numpy as np
import pprint as pp
import json
import lightning as pl

# lactchain imports
from lactchain.generators import generator_factory, VLLMGenerator, VLLMGeneratorConfig
from lactchain.prompts import StrategyPromptTemplate

# base classes
from lactchain.configs.base_config import BaseConfig
from lactchain.classes.base_generator import LLMGenerator
from lactchain.classes.base_prompt import BasePromptTemplate
from lactchain.classes.base_lactchain import LactChain, StrategyChain

_T = TypeVar('_T')


class StrategyChainConfig(BaseConfig):
    '''Base Config for Strategy ChainConfig'''

    generator_config: VLLMGeneratorConfig = Field(
        default_factory=VLLMGeneratorConfig,
        description='generator config'
    )

class ListOfMoves(BaseModel):
    moves: List[str]


class StrategyChain(StrategyChain):
    '''Single Actor-Based Chain for On-Policy Sampling
    For Now: 
    ========

    Initialize: Strategy + Parser
    Input: observation + info
    Output: sequence of actions
    '''

    def __init__(self,
                 generator: LLMGenerator,
                 prompt_template: StrategyPromptTemplate,
                 solver: Optional[Callable]=None
                 ) -> None:
        '''Container actor chain class'''

        self.generator = generator
        self.prompt_template = prompt_template
        self.solver = solver
        
    def _preprocess(self, environment_or_task: str | list[str]) -> list[str]: 
        '''Returns processed environments_or_task'''
        prompts = self.prompt_template.preprocess(environment_or_task=environment_or_task)
        return prompts

    def sample_strategies(self, environment_or_task: str | list[str]) -> list[str]:
        '''Samples strategies in a list'''
        
        prompts = self._preprocess(environment_or_task)
        outputs = self.generator.generate(prompts)
        
        return outputs

