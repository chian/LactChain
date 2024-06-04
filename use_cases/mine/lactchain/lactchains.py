import sys, os
from langchain_community.llms import Ollama
from textwrap import dedent
from typing import Any, List, Union, Dict, Optional, Literal
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPT2LMHeadModel, AutoModelForCausalLM
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import torch, torch.nn as nn, torch.nn.functional as F
import pprint as pp
sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')
from classes.lactchain import LactChain, Context, Component
from langchain_core.prompts import PromptTemplate
from use_cases.mine.lactchain.config import BaseConfig
from use_cases.mine.lactchain.langchain_backend import LangChainGenerator, GeneratorConfig
from use_cases.mine.lactchain.vllm_backend import VLLMGenerator, VLLMGeneratorConfig
from use_cases.mine.lactchain.huggingface_backend import HuggingFaceGenerator, HuggingFaceGeneratorConfig
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
STRATEGY_PROMPT=dedent("""\
            You are an intelligent strategist agent that is in gridworld. 
            Come up with a plausable strategy for how you might want to navigate gridworld and 
            help you reach the goal. Your response must be some kind of move, even if you have to guess. 
            """)

PROMPT_TEMPLATE=dedent("""\
            There are only 2 types of moves you can make:
            
            1. move forward
            2. turn left
            
            Come up with a combination of those two moves in order
            to successfully carry out the action: {strategy}
            
            Your final answer should be in the format of a python list 
            of moves, where each move is one of the 2 types listed above.
            E.g. ['move forward', 'turn left']. 
                                              
            Here is your current position in grid world: {position}
            """)

class Strategy(object): 
    def __init__(self, 
                 prompt_template:str, 
                 strategy_prompt:str, 
                 ): 
        self.strategy_prompt=strategy_prompt
        self.prompt_template=prompt_template

    @staticmethod
    def show_state(state:Dict[str, Any]): 
        pp.print(state)
    
    def compile_policy_prompt(self, state:Dict[str, Any], strategy:str) -> str: 
        strategy=self.prompt_template.format(strategy=strategy, position=state)
        return strategy
    
    def modify_strategy_prompt(self, new_strategy_prompt:str) -> str:
        self.strategy_prompt=new_strategy_prompt
        return f'New strategy prompt is:\n{self.new_strategy}'

'''Idea: have the strategy be a learnable prefix token you append in the prompt'''

class MyLactChain(nn.Module): 
    def __init__(self,
                 config:PolicyConfig,
                 prompt_template:str, 
                 strategy_prompt:str, 
                 model:str, 
                 cache_dir:str
                 ): 
        super().__init__()
        '''We want the llm to output strategy prompt, and then the actual action'''
        self.strategy=Strategy(prompt_template, strategy_prompt)

        backends={
            'langchain':LangChainGenerator,
            'vllm':VLLMGenerator,
            'huggingface':HuggingFaceGenerator
            }
        
        _generator=backends.get(config.backend)

        if config.backend=='langchain': 
            config.langchainconfig.model=model
            self.generator=_generator(**config.langchainconfig.model_dump())
        elif config.backend=='huggingface': 
            config.huggingfaceconfig.pretrained_model_name_or_path=model
            self.generator=_generator(config.huggingfaceconfig)
        elif config.backend=='vllm': 
            config.vllmconfig.llm_name=model
            self.generator=_generator(config.backend)(**config.vllmconfig.model_dump())

        if self.generator.tokenizer.pad_token is None: 
            self.generator.tokenizer.pad_token=self.generator.tokenizer.eos_token

        self.pydantic_parser = PydanticOutputParser(pydantic_object=ListOfMoves)
        self.format_instructions = self.pydantic_parser.get_format_instructions()

    def compile_strategy(self, ): 
        ...

    def propose_strategy(self, strategy_prompt:str) -> str: 
        
        ...
    

    def propose_strategy(self, new_strategy_prompt:Optional[str]=None) -> str:
        self.strategy.modify_strategy_prompt(new_strategy_prompt) if new_strategy_prompt else None
        prompt=PromptTemplate(template="{input}", 
                              input_variables=['input'], 
                              partial_variables={"format_instructions": self.format_instructions})
        chain=prompt | self.pipeline_model | self.pydantic_parser
        output=chain.invoke({'input':self.strategy.strategy_prompt})
        breakpoint()
        strategy=self.pydantic_parser.parse(output)
        return strategy
    
    def sample_action(self, state:Dict[str, Any], strategy:str) -> List[str]: 
        policy=self.strategy.compile_policy_prompt(state)
        input_ids, _attention_mask=self.tokenizer(policy, **self.tokenizer_configs)
        policy_tokens=self.model.generate(input_ids)
        action=self.tokenizer.decode(policy_tokens.squeeze(), skip_special_tokens=True)
        return ...



if __name__=="__main__": 

    ####### Automodel test #######
    policy_config=PolicyConfig()
    lactchain=MyLactChain(policy_config, PROMPT_TEMPLATE, STRATEGY_PROMPT, 
                          'microsoft/Phi-3-mini-4k-instruct', './')
    
    prompts=['Hello how are you doing?', 'Can you write a poem for me?']
    # breakpoint()
    output=lactchain.generator.generate(prompts)
    breakpoint()