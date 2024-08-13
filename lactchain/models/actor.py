from textwrap import dedent
from typing import Any, List, Dict, Optional, Literal, Tuple, TypeVar, Callable
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import torch
import numpy as np
import pprint as pp
import json
import lightning as pl
from lactchain.configs.base_config import BaseConfig

from lactchain.classes.base_generator import LLMGenerator
from lactchain.classes.base_prompt import BasePromptTemplate
from lactchain.classes.base_lactchain import LactChain, StrategyChain

from lactchain.models.backends.langchain_backend import LangChainGenerator, GeneratorConfig
from lactchain.models.backends.vllm_backend import VLLMGeneratorConfig, VLLMGenerator
from lactchain.models.backends.huggingface_backend import (HuggingFaceGenerator, 
                                                           HuggingFaceGeneratorConfig, 
                                                           LoraConfigSettings)
from lactchain.models.prompts import Prompts
############################################################################

_T = TypeVar['_T']

class ActorConfig(BaseConfig):
    backend:Literal['langchain', 'huggingface', 'vllm']=Field('huggingface')
    model:str=Field('mistralai/Mixtral-8x7B-Instruct-v0.1')
    langchainconfig:GeneratorConfig=Field(default_factory=GeneratorConfig)
    huggingfaceconfig:HuggingFaceGeneratorConfig=Field(default_factory=HuggingFaceGeneratorConfig)
    vllmconfig:VLLMGeneratorConfig=Field(default_factory=VLLMGeneratorConfig)

class ListOfMoves(BaseModel):
    moves: List[str]

STRATEGY=dedent("""\
            You are an intelligent strategist agent that is in gridworld.
            Come up with a plausable strategy for how you might want to navigate gridworld and
            help you reach the goal. Your response must be some kind of move, even if you have to guess.
            """)

PROMPT_TEMPLATE=dedent("""\
            <s>[INST]
            There are only 2 types of moves you can make:

            1. move forward
            2. turn left

            Come up with a combination of those two moves in order
            to successfully carry out the task:
            {strategy}

            Your final answer should be in the format of a python list
            of moves, where each move is one of the 2 types listed above.
            E.g. ["move forward", "turn left"]. DO NOT CHOOSE ANY OTHER TYPES OF MOVES
            OR YOU WILL BE PUNISHED

            All of your output must be stored in a json in the following format, and nothing else:
            {{
            "explain": "// Your explanation and logic goes here //"
            "moves": "// Your sequence of moves goes here //"
            }}
            YOU ARE NOT ALLOWED TO OUTPUT ANYTHING ELSE THAT DOES NOT STRICTLY ADHERE TO THE JSON FORMAT ABOVE.
            TAKE NOTE THAT THE KEYS IN YOUR JSON OUTPUT SHOULD BE IN DOUBLE QUOTES
            
            An example of a correctly formatted output is this: 
            
            {{
            "explain": "Since the grid size is 4 and the goal is at (4, 4) we need to move towards that bottom right position"
            "moves": ["move forward", "turn left"]
            }}

            Here is your current position in grid world: 
            {position}
            Here is some extra information of grid world: 
            {info}
            
            [/INST]
            """)

class Strategy(object):
    def __init__(self,
                 prompt_template:str,
                 strategy:str,
                 ):
        self.strategy=strategy
        self.prompt_template=prompt_template

    @staticmethod
    def show_state(state:Dict[str, Any]):
        pp.print(state)

    def __call__(self, state:Dict[str, Any], info:str) -> str:
        '''formats final prompt from state input from env and strategy declaration'''
        final_prompt=self.prompt_template.format(strategy=self.strategy, position=state, info=info)
        return final_prompt

    def modify_strategy_prompt(self, new_strategy:str) -> str:
        '''modify the strategy input'''
        self.strategy=new_strategy
        return f'New strategy prompt is:\n{self.strategy}'

class LactChain(object):
    
    MODEL_MAP:dict[str, str]={
        'meta-llama/Meta-Llama-3-8B-Instruct': 'llama-3', 
        'meta-llama/Meta-Llama-3-8B':'llama-3',
        'mistralai/Mistral-7B-v0.3': 'mistral-7b', 
        'mistralai/Mixtral-8x7B-Instruct-v0.1':'mistral-7b'
        }
    
    def __init__(self,
                 backend:str,
                 model:str,
                 model_type:str,
                 config:ActorConfig,
                 lora_config:Optional[LoraConfigSettings]=None, 
                 ):
        super().__init__()
        '''We want the llm to output strategy prompt, and then the actual action'''
        assert model_type in self.MODEL_MAP.values() , f'''Current supported models are only llama-3 8B models and mistral 7B-V0.3 and Mixtral 8x7B'''
        
        MODEL_TYPE=model_type
        self._prompt=Prompts(backend, MODEL_TYPE)
        
        self._strategy=Strategy(self._prompt.prompt, self._prompt.strategy)
        self.error_number=1000
        self._outputs=''
        
        backends={
            'langchain':LangChainGenerator,
            'vllm':VLLMGenerator,
            'huggingface':HuggingFaceGenerator
            }

        _generator=backends.get(config.backend)

        if config.backend=='langchain':
            config.langchainconfig.model=model
            generator=_generator(**config.langchainconfig.model_dump())
            self.pydantic_parser = PydanticOutputParser(pydantic_object=ListOfMoves)
            self.format_instructions = self.pydantic_parser.get_format_instructions()

        elif config.backend=='huggingface':
            config.huggingfaceconfig.pretrained_model_name_or_path=model
            if lora_config:
                generator=_generator(config.huggingfaceconfig, lora_config)
            else:
                generator=_generator(config.huggingfaceconfig)
                
            for param in generator.model.parameters():
                param.requires_grad = False
                
        elif config.backend=='vllm': 
            config.vllmconfig.pretrained_model_name_or_path=model
            generator=_generator(config.vllmconfig)

        self.generator=generator
        self.config=config
        
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
            # for action in mapped_actions: 
            #     if action not in [0, 1]: 
            #         mapped_actions=np.array([self.error_number])
            #         drop_indices.append(batch_idx)
            #         break   
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
    
    
    
class StrategyChain(StrategyChain):
    '''Single Actor-Based Chain for On-Policy Sampling''' 
    def __init__(self, 
                 generator: LLMGenerator,
                 prompt_template: BasePromptTemplate, 
                 solver: Optional[Callable]
                 ) -> None: 
        '''Container actor chain class'''

        self.generator=generator
        self.prompt_template=prompt_template
        self.solver = solver
        
    
    @classmethod
    def initialize_chain(cls: _T, 
                         generator: LLMGenerator, 
                         prompt_template: BasePromptTemplate
                         ) -> _T: 
        '''Initialize the latchain by passing in the generator and prompt template'''
    
    def sample_actions(self, 
                       states: list[str], 
                       infos: list[str]
                       ) -> list[str]: 
        ...    
        
    def batch_parse_outputs(self, outputs:list[str]) -> list[str]:
        '''Loops through the list of outputs and json parses them to return a list of 
        processed strings
        '''
        parsed_outputs=[]
        for _, output in enumerate(outputs):
            # print(output)
            parsed_outputs.append(json.loads(output))
        return parsed_outputs

if __name__=="__main__":

    # output='''{"explain":"Hlleo", "actions":["move right", "move left"]}'''
    from lactchain.models.prompts import Prompts
    
    ACTOR_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
    
    prompt=Prompts('vllm', model_type='mistral-7b')
    
    lora_config=LoraConfigSettings()
    policy_config=ActorConfig(backend='vllm')
    
    actor=LactChain(backend='vllm',
                    model=ACTOR_PATH, 
                    model_type='mistral-7b',
                    config=policy_config, 
                    lora_config=lora_config)
    

