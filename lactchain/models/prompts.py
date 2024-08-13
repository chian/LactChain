from __future__ import annotations
from dataclasses import dataclass, field
from textwrap import dedent
from enum import Enum
from typing import Literal, Union, Optional

class HuggingFacePrompts(Enum): 
    
    '''Enum class for HuggingFace Prompts'''
    GRIDWORLD_STRATEGY=dedent("""\
                You are an intelligent strategist agent that is in gridworld.
                Come up with a plausable strategy for how you might want to navigate gridworld and
                help you reach the goal. Your response must be some kind of move, even if you have to guess.
                """)
    MISTRAL_7B_TEMPLATE=dedent("""\
                <s>[INST]
                There are only 2 types of moves you can make:

                1. move forward
                2. turn left

                Come up with a combination of those two moves in order
                to successfully carry out the task:
                {strategy}

                Your final answer should be in the format of a python list
                of moves, where each move is one of the 2 types listed: ["move forward", "turn left"]. 
                DO NOT CHOOSE ANY OTHER TYPES OF MOVES OR YOU WILL BE PUNISHED

                All of your output must be stored in a json in the following format, and nothing else:
                {{
                "explain": "// Your explanation and logic goes here //",
                "moves": "// Your sequence of moves goes here //"
                }}
                YOU ARE NOT ALLOWED TO OUTPUT ANYTHING ELSE THAT DOES NOT STRICTLY ADHERE TO THE JSON FORMAT ABOVE.
                TAKE NOTE THAT THE KEYS IN YOUR JSON OUTPUT SHOULD BE IN DOUBLE QUOTES. YOU MUST WRAP YOUR MOVES IN A LIST. 
                
                An example of a correctly formatted output is this: 
                
                {{
                "explain": "Since the grid size is 4 and the goal is at (4, 4) we need to move towards that bottom right position",
                "moves": ["move forward", "turn left"]
                }}
                
                An example of an incorrectly formatted output is this since it is missing a closing bracket for the list of moves, and is missing a comma in between the 'explain' and 'moves' section.
                In addition, an incorrect move "move backward" is selected, which is not a valid move:
                
                {{
                "explain": "Since the grid size is 4 and the goal is at (4, 4) we need to move towards that bottom right position"
                "moves": ["move backward", "turn left"
                }}

                Here is your current position in grid world: 
                {position}
                Here is some extra information of grid world: 
                {info}
                
                [/INST]
                """)
    # prompt format from: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3
    LLAMA_3_TEMPLATE=dedent("""\
                <|begin_of_text|><|start_header_id|>user<|end_header_id|>
                There are only 2 types of moves you can make:

                1. move forward
                2. turn left

                Come up with a combination of those two moves in order
                to successfully carry out the task:
                {strategy}

                Your final answer should be in the format of a python list
                of moves, where each move is one of the 2 types: ["move forward", "turn left"]. 
                DO NOT CHOOSE ANY OTHER TYPES OF MOVES OR YOU WILL BE PUNISHED

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
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """)


class VLLMPrompts(Enum): 
    '''VLLM Prompts'''
    GRIDWORLD_STRATEGY=dedent("""\
                You are an intelligent strategist agent that is in gridworld.
                Come up with a plausable strategy for how you might want to navigate gridworld and
                help you reach the goal. Your response must be some kind of move, even if you have to guess.
                """)
    # prompt format from: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3
    TEMPLATE=dedent("""\
                There are only 2 types of moves you can make:

                1. move forward
                2. turn left

                Come up with a combination of those two moves in order
                to successfully carry out the task:
                {strategy}

                Your final answer should be in the format of a python list
                of moves, where each move is one of the 2 types listed below.
                E.g. ["move forward", "turn left"]. DO NOT CHOOSE ANY OTHER TYPES OF MOVES
                OR YOU WILL BE PUNISHED

                All of your output must be stored in a json in the following format, and nothing else:
                {{
                "explain": "// Your explanation and logic goes here //",
                "moves": "// Your sequence of moves goes here //"
                }}
                YOU ARE NOT ALLOWED TO OUTPUT ANYTHING ELSE THAT DOES NOT STRICTLY ADHERE TO THE JSON FORMAT ABOVE.
                TAKE NOTE THAT THE KEYS IN YOUR JSON OUTPUT SHOULD BE IN DOUBLE QUOTES. YOU MUST WRAP YOUR MOVES IN A LIST. 
                
                An example of a correctly formatted output is this: 
                
                {{
                "explain": "Since the grid size is 4 and the goal is at (4, 4) we need to move towards that bottom right position",
                "moves": ["move forward", "turn left"]
                }}
                
                An example of an incorrectly formatted output is this since it is missing a closing bracket for the list of moves, and is missing a comma in between the 'explain' and 'moves' section: 
                {{
                "explain": "Since the grid size is 4 and the goal is at (4, 4) we need to move towards that bottom right position"
                "moves": ["move forward", "turn left"
                }}

                Here is your current position in grid world: 
                {position}
                Here is some extra information of grid world: 
                {info}
                """)
        
@dataclass
class Prompts:
    '''Class that maps model types to prompt templates that contain the right tokens'''
    PROMPT_MAPPING = {
        'llama-3': 'LLAMA_3_TEMPLATE',
        'mistral-7b': 'MISTRAL_7B_TEMPLATE',
    }
    
    BACKEND={
        'vllm':VLLMPrompts, 
        'huggingface':HuggingFacePrompts
    }
    
    backend:Literal['vllm', 'huggingface']
    model_type:Literal['llama-3', 'mistral-7b']
    
    prompt:str=field(init=False) # field that depends on template_type, so init=False 
    strategy:str=field(init=False)
    
    def __post_init__(self):
        prompt_class=self.BACKEND.get(self.backend)
        self.strategy=prompt_class.GRIDWORLD_STRATEGY.value
        
        if self.backend=='huggingface': 
            self.prompt=getattr(prompt_class, self.PROMPT_MAPPING.get(self.model_type)).value
            
        elif self.backend=='vllm':
            self.prompt=prompt_class.TEMPLATE.value
            
            
###################### TODO: Split Prompts --> Strategy Prompt + Solver Prompt ###################
# L1: Strategy Prompt --> Strategey ||| L2: Strategy + Output Formattter --> Action

class BaseStrategies(Enum): 
    '''Stores input prompt strategies'''
    GRIDWORLD_STRATEGY=dedent("""\
            You are an intelligent strategist agent that is in gridworld.
            Come up with a plausable strategy for how you might want to navigate gridworld and
            help you reach the goal. Your response must be some kind of move, even if you have to guess.
            """)
    WEI_STRATEGY=...
    
@dataclass 
class StrategyPrompt: 
    
    '''template for holding strategy'''
    template:str=field(
        default='{input}'
    )
    
    '''Input string for strategy; set to BaseStrategy as default'''
    strategy:Union[BaseStrategies, str]=field(
        default=BaseStrategies.GRIDWORLD_STRATEGY.value
    )
    
    def __post_init__(self):
        prompt_class=self.BACKEND.get(self.backend)
        self.strategy=prompt_class.GRIDWORLD_STRATEGY.value
        
        if self.backend=='huggingface': 
            self.prompt=getattr(prompt_class, self.PROMPT_MAPPING.get(self.model_type)).value
            
        elif self.backend=='vllm':
            self.prompt=prompt_class.TEMPLATE.value
        
    
        
if __name__=="__main__": 
    
    prompt=Prompts('huggingface', 'llama-3')
    
    breakpoint()