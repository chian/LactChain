import sys, os
from langchain_community.llms import Ollama
from textwrap import dedent
from typing import Any, List, Union, Dict, Optional, Literal, Tuple
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPT2LMHeadModel, AutoModelForCausalLM
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import torch, torch.nn as nn, torch.nn.functional as F
import pprint as pp
import json
from peft import prepare_model_for_kbit_training, LoraModel, LoraConfig
sys.path.append(os.getcwd()+'/../../../')
from classes.lactchain import LactChain, Context, Component
from langchain_core.prompts import PromptTemplate
from use_cases.mine.lactchain.config import BaseConfig
from use_cases.mine.lactchain.langchain_backend import LangChainGenerator, GeneratorConfig
from use_cases.mine.lactchain.vllm_backend import VLLMGenerator, VLLMGeneratorConfig
from use_cases.mine.lactchain.huggingface_backend import HuggingFaceGenerator, HuggingFaceGeneratorConfig
from use_cases.mine.lactchain.my_huggingface_backend import MyHuggingFaceGenerator, LoraConfigSettings
############################################################################

class PolicyConfig(BaseConfig):
    backend:Literal['langchain', 'huggingface', 'vllm']=Field('huggingface')
    model:str=Field('mistralai/Mixtral-8x7B-Instruct-v0.1')
    langchainconfig:GeneratorConfig=Field(default_factory=GeneratorConfig)
    huggingfaceconfig:HuggingFaceGeneratorConfig=Field(default_factory=HuggingFaceGeneratorConfig)
    vllmconfig:VLLMGeneratorConfig=Field(default_factory=VLLMGeneratorConfig)

class ListOfMoves(BaseModel):
    moves: List[str]

'''Learn strategy via contrastive method:
1.) Have a big model and small model make the guess, and
'''
ORIGINAL_STRATEGY="""\
            You are in gridworld. Make a move to help you reach the goal.
            Your response must be some kind of move, even if you have to guess.
            """
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

class MyLactChain(nn.Module):
    def __init__(self,
                 model:str,
                 config:PolicyConfig,
                 lora_config:Optional[LoraConfigSettings]=None
                 ):
        super().__init__()
        '''We want the llm to output strategy prompt, and then the actual action'''
        self._strategy=Strategy(PROMPT_TEMPLATE, STRATEGY)

        backends={
            'langchain':LangChainGenerator,
            'vllm':VLLMGenerator,
            'huggingface':MyHuggingFaceGenerator
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

    def compile_prompt(self, state:str, info:str) -> str:
        return self._strategy(state, info)

    def parse_outputs(self, outputs:list[str]) -> list[str]:
        try:
            return [json.loads(output) for output in outputs]
        except Exception as e:
            return dedent(f'''Your output string is not correctly formatted for {pp.pformat(outputs)}.
                            Here is the error{e}''')

    def batch_parse_outputs(self, outputs:list[str]) -> list[str]:
        parsed_outputs=[]
        for i, output in enumerate(outputs):
            try:
                parsed_outputs.append(json.loads(output))
            except Exception as e:
                print(f'PARSING ERROR FOR ELEMENT {i} IN BATCH...SKIPPING')
                pass
        return parsed_outputs

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
        action=parsed_outputs[0]['moves'] # 0 since we are assuming list of actions is just [action]
        context=parsed_outputs[0]['explain']
        return action, context

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
        return actions, contexts

if __name__=="__main__":

    output='''{"explain":"Hlleo", "actions":["move right", "move left"]}'''

    policy_config=PolicyConfig()
    lactchain=MyLactChain(policy_config, "mistralai/Mistral-7B-Instruct-v0.3", './')
    states=[
        {'x':10, 'y':5, 'orientation':'right'},
        {'x':3, 'y':5, 'orientation':'left'}
        ]
    info=[
        'grid world is size 15, 15',
        'grid world is size 40x10'
        ]
    outputs=lactchain.sample_actions(states, info)

    breakpoint()
