from __future__ import annotations

'''Class for Implementing Action Mapping Prompts'''

from dataclasses import dataclass, field
from textwrap import dedent
from enum import Enum
from typing import Literal, Union, Optional, Any, Tuple
import numpy as np

from lactchain.configs.base_config import BaseConfig

class ActionsPromptTemplateConfig(BaseConfig):
    """Configuration for the StrategyPromptTemplate"""
    name: Literal['action'] = 'action'  # type: ignore[assignment]
    solver: Literal['gridworld'] = 'gridworld'
    
class ActionSolver: 
    '''Class used to map llm outputs to actions for environment'''
    
    action_map: dict[str, Any] = {
        'move forward':0, 
        'turn left':1
        }
    
    def convert(self, batch_moves: str | list[str]) -> np.ndarray: 
        '''Converts a batch of action strings to their array actions'''

        if isinstance(batch_moves, str): 
            batch_moves = [batch_moves]

        batch_mapped_actions=[]
        for idx, moves in enumerate(batch_moves):
            mapped_actions=np.array(
                [self.action_map.get(move) for move in moves]
                )
            for action in mapped_actions: 
                assert action in [0, 1], f'MAP ACTION ERROR: {action} AT {idx}, ACTION MUST BE [0, 1]'
            batch_mapped_actions.append(mapped_actions)
            
        return batch_mapped_actions
    
class ActionPromptTemplate:
    """Question answer prompt template."""

    template_with_context: str = dedent("""\
                <s>[INST]
                There are only 2 types of moves you can make:

                1. move forward
                2. turn left

                Come up with a combination of those two moves in order
                to successfully carry out the task. 
                
                Below is a general strategy for how to think about the problem:
                {strategy}

                Your final answer should be in json format, where your actions are formatted as a python list
                of moves, where each move is one of the 2 types listed below.
                E.g. ["move forward", "turn left"]. DO NOT CHOOSE ANY OTHER TYPES OF MOVES
                OR YOU WILL BE PUNISHED

                YOUR OUTPUT MUST ONLY BE IN THE FOLLOWING JSON FORMAT:
                {{
                "explain": "// Your explanation and logic goes here //",
                "moves": "// Your sequence of moves goes here //"
                }}
                YOU ARE NOT ALLOWED TO OUTPUT ANYTHING ELSE THAT DOES NOT STRICTLY ADHERE TO THE JSON FORMAT ABOVE.
                YOU CAN ONLY OUTPUT JSON.
                TAKE NOTE THAT THE KEYS IN YOUR JSON OUTPUT SHOULD BE IN DOUBLE QUOTES. YOU MUST WRAP YOUR MOVES IN A LIST. 
                
                EXAMPLE OF CORRECTLY FORMATTED OUTPUT:
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
                {state}
                Here is some extra information of grid world: 
                {info}
                [/INST]
                """)
        

    template_no_context: str = dedent("""\
                <s>[INST]
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
                
                Since you have no information, make an educated guess!
                [/INST]
                """)
        

    def __init__(self, config: ActionsPromptTemplateConfig) -> None:
        """Initialize the QuestionAnswerPromptTemplate."""
        self.config = config

    def _format_prompt(
        self,
        strategy:str | list[str], 
        state:Optional[str | list[str]], 
        info:Optional[str | list[str]]
        ) -> str:
        """Format the prompt with the question and context."""
        
        return self.template_with_context.format(
            strategy=strategy, 
            state=state, 
            info=info
        )

    def preprocess(
        self,
        strategy:str | list[str], 
        state:Optional[str | list[str]], 
        info:Optional[str | list[str]]
        ) -> list[str]:
        """Preprocess the text into prompts.

        Parameters
        ----------
        text : str
            The text to format.

        Returns
        -------
        list[str]
            The formatted prompts.
        """
        # Ensure text is a list
        if isinstance(strategy, str):
            strategy = [strategy]
            state = [state]
            info = [info]
            
        # If no contexts are provided, use the no-context template
        if state is None or info is None:
            return list(map(self._format_prompt, strategy)) * len(strategy)

        # Build the prompts using the template
        return list(map(self._format_prompt, strategy, state, info))

    def postprocess(self, responses: list[str]) -> list[str]:
        """Postprocess the responses.

        Parameters
        ----------
        responses : list[str]
            The responses to postprocess.

        Returns
        -------
        list[str]
            The postprocessed responses.
        """
        # If present, remove the option number from the response
        responses = [
            r[3:] if r[:2] in ['1.', '2.', '3.', '4.'] else r
            for r in responses
        ]
        # If present, remove the period from the end of the response
        responses = [r if r and r[-1] != '.' else r[:-1] for r in responses]

        # Cast responses to lower caps in case model capitalized answers.
        responses = [r.lower() for r in responses]

        return responses
    