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
from lactchain.prompts import ActionPromptTemplate, ActionsPromptTemplateConfig, ActionSolver
# base classes 
from lactchain.configs.base_config import BaseConfig
from lactchain.classes.base_generator import LLMGenerator
from lactchain.classes.base_prompt import BasePromptTemplate
from lactchain.classes.base_lactchain import LactChain, ActorChain

_T = TypeVar('_T')

class ActorChainConfig(BaseConfig):
    '''Base Config for ActorChainConfig'''
    
    generator_config: VLLMGeneratorConfig = Field(
        default_factory=VLLMGeneratorConfig, 
        description='generator config'
        )
    
class ListOfMoves(BaseModel):
    moves: List[str]

class ActorChain(ActorChain):
    '''Single Actor-Based Chain for On-Policy Sampling
    For Now: 
    ========
    
    Initialize: Strategy + Parser
    Input: observation + info
    Output: sequence of actions
    ''' 
    def __init__(self, 
                 generator: LLMGenerator,
                 prompt_template: ActionPromptTemplate, 
                 solver: Optional[ActionSolver]=None, 
                 environment_processor: Optional[Callable]=None
                 ) -> None: 

        '''Container actor chain class'''

        self.generator=generator
        self.prompt_template=prompt_template
        self.solver = solver        
        self.environment_processor = environment_processor
        
    def _preprocess(self, 
                    strategies: str | list[str], 
                    states: str | list[str], 
                    infos: Optional[str | list[str]] = None
                    ) -> list[str]: 
        '''Preprocesses strings into prompt templates'''
        
        if self.environment_processor: 
            states, infos = self.environment_processor(states, infos)
        
        prompts = self.prompt_template.preprocess(strategy=strategies, 
                                                  state=states, 
                                                  info=infos)
            
        return prompts
        
    def parse_outputs(self, outputs:list[str]) -> list[str]:
        '''Loops through the list of outputs and json parses them to return a list of 
        processed strings
        '''
        parsed_outputs=[]
        for _, output in enumerate(outputs):
            parsed_outputs.append(json.loads(output))
            
        return parsed_outputs
        
    def map_actions(self, batch_actions:list[str]) -> list[np.ndarray]: 
        '''Helper function that maps list of processed outputs from llm into binary actions 
        as a list of arrays
        '''
        
        batch_mapped_actions = self.solver.convert(batch_actions)
            
        return batch_mapped_actions
    
    def sample_actions(self,
                       strategies: str | list[str], 
                       states: str | list[str],
                       infos: str | list[str]
                       ) -> list[np.ndarray]:
        '''Samples batches of actions as List[array(int={0, 1})] where list is the batch, 
        array is the compound actions'''
        
        prompts = self._preprocess(strategies=strategies, 
                                   states=states, 
                                   infos=infos)
        
        outputs = self.generator.generate(prompts)
        parsed = self.parse_outputs(outputs)
        actions = self.map_actions(parsed)
        
        return actions
    
    