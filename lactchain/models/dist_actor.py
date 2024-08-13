from textwrap import dedent
from typing import Any, List, Dict, Optional, Literal, Tuple
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import torch
import numpy as np
import pprint as pp
import json
import lightning as pl
from lactchain.configs.base_config import BaseConfig
from lactchain.models.backends.langchain_backend import LangChainGenerator, GeneratorConfig
from lactchain.models.backends.vllm_backend import VLLMGeneratorConfig, VLLMGenerator
from lactchain.models.backends.huggingface_backend import (HuggingFaceGenerator, 
                                                           HuggingFaceGeneratorConfig, 
                                                           LoraConfigSettings)
from lactchain.models.prompts import Prompts
############################################################################

class ActorConfig(BaseConfig):
    backend:Literal['huggingface', 'vllm']=Field(
        'huggingface'
        )
    model:str=Field(
        'mistralai/Mixtral-8x7B-Instruct-v0.1'
        )
    huggingfaceconfig:HuggingFaceGeneratorConfig=Field(
        default_factory=HuggingFaceGeneratorConfig
        )
    vllmconfig:VLLMGeneratorConfig=Field(
        default_factory=VLLMGeneratorConfig
        )

class ListOfMoves(BaseModel):
    moves: List[str]
    
    
MODEL_MAP:dict[str, str]={
    'meta-llama/Meta-Llama-3-8B-Instruct': 'llama-3', 
    'meta-llama/Meta-Llama-3-8B':'llama-3',
    'mistralai/Mistral-7B-v0.3': 'mistral-7b', 
    'mistralai/Mixtral-8x7B-Instruct-v0.1':'mistral-7b'
    }

class SubChain: 
    def __init__(): 
        ...




    
class LactChain(object):
    '''Class for Actor Strategy Inference'''
    def __init__(self,
                 strategy:Strategy,
                 pretrained_model_name_or_path:str,
                 model_type:str, 
                 generator_config:HuggingFaceGeneratorConfig | VLLMGeneratorConfig, 
                 lora_config:Optional[LoraConfigSettings]=None
                 ):
        super().__init__()
        '''We want the llm to output strategy prompt, and then the actual action'''

        backends={
            'vllm':VLLMGenerator,
            'huggingface':HuggingFaceGenerator
            }

        _generator=backends.get(generator_config.backend)

        if generator_config.backend=='huggingface':
            generator_config.huggingfaceconfig.pretrained_model_name_or_path=pretrained_model_name_or_path
            if lora_config:
                generator=_generator(generator_config.huggingfaceconfig, lora_config)
            else:
                generator=_generator(generator_config.huggingfaceconfig)
            for param in generator.model.parameters():
                param.requires_grad = False
                
        elif generator_config.backend=='vllm': 
            generator_config.vllmconfig.pretrained_model_name_or_path=pretrained_model_name_or_path
            generator=_generator(generator_config.vllmconfig)

        # fix strategies
        self.generator=_generator
        self.config=generator_config
        
        self._strategy=strategy
        self._error_number=1000
        self._outputs=''
        
    @property
    def strategy(self) -> None: 
        return self._strategy
        
    @property
    def outputs(self): 
        '''
        Property that returns the list of outputs output by a language model as a str
        '''
        return self._outputs

    def compile_prompts(self, 
                        state:str | list[str], 
                        info:str | list[str]
                        ) -> str | list[str]:
        '''Returns the whole prompt as a str. You can pass in a list of states and infos to get 
        a batch output of strategies 
        '''
        if isinstance(state, list) and isinstance(info, list): 
            strategies=[]
            for state, info in zip(state, info): 
                strategies.append(self._strategy(state, info))
            return strategies
        else: 
            return self._strategy(state, info)
            
    def map_actions(self, 
                    batch_actions:list[str]
                    ) -> Tuple[list[np.ndarray], list[int]]: 
        '''Helper function that maps list of processed outputs from llm into binary actions 
        as a list of arrays
        '''
        map={
            'move forward':0, 
            'turn left':1
        }
        batch_mapped_actions=[]
        drop_indices=[]

        for batch_idx, actions in enumerate(batch_actions):
            mapped_actions=np.array([map.get(action) for action in actions])
            for action in mapped_actions: 
                assert action in [0, 1, 1000], f'MAP ACTION ERROR: {action} AT {batch_idx}, ACTION MUST BE [0, 1, 1000]'
            batch_mapped_actions.append(mapped_actions)
            
        return batch_mapped_actions, drop_indices

    def batch_parse_outputs(self, outputs:list[str]) -> list[str]:
        '''Loops through the list of outputs and json parses them to return a list of 
        processed strings
        '''
        parsed_outputs=[]
        for _, output in enumerate(outputs):
            # print(output)
            parsed_outputs.append(json.loads(output))
            
        return parsed_outputs

    @torch.inference_mode()
    def sample_actions(self,
                       states:Dict[str, Any],
                       infos:str
                       ) -> Tuple[list[str], str]:
        '''Samples batches of actions as List[array(int={0, 1})] where list is the batch, 
        array is the compound actions'''
        strategies=[]
        states=[states] if isinstance(states, dict) else states
        infos=[infos] if isinstance(infos, str) else infos
        for (state, info) in zip(states, infos):
            strategy=self._strategy(state, info)
            strategies.append(strategy)
        
        batch_size=len(strategies) # optional batch size 
        if self.config.backend=='vllm':
            outputs=self.generator.generate(strategies)
        elif self.config.backend=='huggingface':
            outputs=self.generator.generate(strategies, batch_size)
        self._outputs=outputs
        
        # print(f'Outputs: {outputs}')
        
        parsed_outputs=self.batch_parse_outputs(outputs)
        actions=[parsed_output['moves'] for parsed_output in parsed_outputs]
        contexts=[parsed_output['explain'] for parsed_output in parsed_outputs]
        
        # print(f'Actions: {actions}')
        
        mapped_actions, num_actions_dropped=self.map_actions(actions)
        
        return mapped_actions, actions, contexts, num_actions_dropped
        
        
        
        
