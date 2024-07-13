from __future__ import annotations
from dataclasses import dataclass, field
from textwrap import dedent

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
            of moves, where each move is one of the 2 types listed above.
            E.g. ["move forward", "turn left"]. DO NOT CHOOSE ANY OTHER TYPES OF MOVES
            OR YOU WILL BE PUNISHED

            All of your output must be stored in a json in the following format, and nothing else:
            {{
            "explain": "// Your explanation and logic goes here //"
            "moves": "// Your sequence of moves goes here //"
            }}
            YOU ARE NOT ALLOWED TO OUTPUT ANYTHING ELSE THAT DOES NOT STRICTLY ADHERE TO THE JSON FORMAT ABOVE.
            TAKE NOTE THAT THE KEYS IN YOUR JSON OUTPUT SHOULD BE IN DOUBLE QUOTES. YOU MUST WRAP YOUR MOVES IN A LIST. 
            
            An example of a correctly formatted output is this: 
            
            {{
            "explain": "Since the grid size is 4 and the goal is at (4, 4) we need to move towards that bottom right position"
            "moves": ["move forward", "turn left"]
            }}
            
            An example of an incorrectly formatted output is this since it is missing a closing bracket for the list of moves: 
            {{
            "explain": "Since the grid size is 4 and the goal is at (4, 4) we need to move towards that bottom right position"
            "moves": ["move forward", "turn left"
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
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """)


        
@dataclass
class Prompts:
    '''Class that maps model types to prompt templates that contain the right tokens'''
    PROMPT_MAPPING = {
        'llama-3': LLAMA_3_TEMPLATE,
        'mistral-7b': MISTRAL_7B_TEMPLATE,
    }
    
    STRATEGY_MAPPING = {
        'gridworld':GRIDWORLD_STRATEGY
    }
    
    model_type: str
    prompt_template: str = field(init=False) # field that depends on template_type, so init=False 
    
    strategy_type: str
    strategy_template: str = field(init=False)
    
    def __post_init__(self):
        self.prompt = self.PROMPT_MAPPING.get(self.model_type)
        self.strategy = self.STRATEGY_MAPPING.get(self.strategy_type)
        
if __name__=="__main__": 
    
    prompt=Prompts('mistral')
    
    breakpoint()