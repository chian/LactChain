from textwrap import dedent
from typing import Any, List, Dict, Optional, Literal, Tuple
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import torch
import numpy as np
import torch.nn as nn
import pprint as pp
import json
import lightning as pl
from lactchain.configs.base_config import BaseConfig
from lactchain.models.backends.langchain_backend import LangChainGenerator, GeneratorConfig
from lactchain.models.backends.vllm_backend import VLLMGeneratorConfig, VLLMGenerator
from lactchain.models.backends.huggingface_backend import (HuggingFaceGenerator, 
                                                           HuggingFaceGeneratorConfig, 
                                                           LoraConfigSettings)
############################################################################

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
            "moves": // Your sequence of moves goes here //
            }}
            YOU ARE NOT ALLOWED TO OUTPUT ANYTHING ELSE THAT DOES NOT STRICTLY ADHERE TO THE JSON FORMAT ABOVE.
            TAKE NOTE THAT THE KEYS IN YOUR JSON OUTPUT SHOULD BE IN DOUBLE QUOTES

            Here is your current position in grid world: {position}
            Here is some extra information of grid world: {info}
            [INST]
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

'''Idea: have the strategy be a learnable prefix token you append in the prompt'''

class LactChain(nn.Module):
    def __init__(self,
                 model:str,
                 config:ActorConfig,
                 lora_config:Optional[LoraConfigSettings]=None, 
                 ):
        super().__init__()
        '''We want the llm to output strategy prompt, and then the actual action'''
        self._strategy=Strategy(PROMPT_TEMPLATE, STRATEGY)

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

        self.generator=generator
        
    @classmethod
    def load_from_checkpoint(cls, checkpoint:str, config:ActorConfig, lora_config:LoraConfigSettings): 
        actor=cls(checkpoint, config, lora_config)
        return actor
    
    def save_model(self, save_path:str): 
        self.save_pretrained(save_path, from_pt=True) 

    def compile_prompt(self, state:str, info:str) -> str:
        return self._strategy(state, info)

    def parse_outputs(self, outputs:list[str]) -> list[str]:
        try:
            return [json.loads(output) for output in outputs]
        except Exception as e:
            return dedent(f'''Your output string is not correctly formatted for {pp.pformat(outputs)}.
                            Here is the error{e}''')
            
    def map_actions(self, batch_actions:list[str]): 
        map={
            'move forward':0, 
            'turn left':1
        }
        batch_mapped_actions=[]
        for actions in batch_actions:
            mapped_actions=np.array([map.get(action) for action in actions])
            batch_mapped_actions.append(mapped_actions)
        return batch_mapped_actions

    def batch_parse_outputs(self, outputs:list[str]) -> list[str]:
        parsed_outputs=[]
        for i, output in enumerate(outputs):
            # try:
            parsed_outputs.append(json.loads(output))
            # except Exception as e:
            #     print(f'PARSING ERROR FOR ELEMENT {i} IN BATCH...SKIPPING')
            #     pass
        return parsed_outputs

    @torch.no_grad()
    def sample_action(self,
                      states:Dict[str, Any],
                      infos:str
                    ) -> Tuple[list[str], str]:
        strategies=[]
        states=[states] if isinstance(states, dict) else states
        infos=[infos] if isinstance(infos, str) else infos
        for (state, info) in zip(states, infos):
            strategy=self._strategy(state, info)
            strategies.append(strategy)

        outputs=self.generator.generate(strategies)
        parsed_outputs=self.parse_outputs(outputs)
        
        actions=self.map_actions(parsed_outputs)
        breakpoint()
        print(outputs)
        action=parsed_outputs[0]['moves'] # 0 since we are assuming list of actions is just [action]
        context=parsed_outputs[0]['explain']
        return action, context

    @torch.no_grad()
    def sample_actions(self,
                      states:Dict[str, Any],
                      infos:str
                    ) -> Tuple[list[str], str]:
        strategies=[]
        states=[states] if isinstance(states, dict) else states
        infos=[infos] if isinstance(infos, str) else infos
        for (state, info) in zip(states, infos):
            strategy=self._strategy(state, info)
            strategies.append(strategy)
        outputs=self.generator.generate(strategies)
        parsed_outputs=self.batch_parse_outputs(outputs)
        actions=[parsed_output['moves'] for parsed_output in parsed_outputs]
        contexts=[parsed_output['explain'] for parsed_output in parsed_outputs]
        mapped_actions=self.map_actions(actions)
        return mapped_actions, actions, contexts
    
        


if __name__=="__main__":

    output='''{"explain":"Hlleo", "actions":["move right", "move left"]}'''

    lora_config=LoraConfigSettings()
    policy_config=ActorConfig()
    
    actor=LactChain('/nfs/lambda_stor_01/homes/bhsu/huggingface_models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de', 
                    policy_config, 
                    lora_config)
    

    breakpoint()