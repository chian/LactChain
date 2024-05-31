import sys, os
from langchain_community.llms import Ollama
from textwrap import dedent
from typing import Any, List, Union, Dict
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.append('/lus/eagle/projects/FoundEpidem/bhsu/2024_research/LactChain')
from classes.lactchain import LactChain, Context, Component

CACHE_PATH=os.getcwd()


class ListOfMoves(BaseModel):
    moves: List[str]


class LactChainA(LactChain):
    def __init__(self, llm):
        super().__init__()
        self.context = {'feedback': [], 'action_choices': []}
        self.llm = llm
        self.message = ""
        self.action_choices = []
        
        self.strategy_component = DeclareStrategy(self.llm, self)
        self.add_component(self.strategy_component)
        
        self.converted_strategy_component = ConvertStrategyToAction(self.llm, self)
        self.add_component(self.converted_strategy_component)

    def update_context(self, key, value):
        if key in self.context:
            self.context[key].append(value)
        else:
            self.context[key] = [value]

    def add_feedback(self, message):
        self.update_context('feedback', message)

    def add_action(self, action):
        self.update_context('action_choices', action)

    def propose_action(self):
        self.strategy_component.execute(self.context)
        self.converted_strategy_component.execute(self.context)
        return self.action_choices


class DeclareStrategy(Component):
    def __init__(self, llm, parent):
        self.llm = llm
        self.parent = parent
        self.move_prompt = dedent("""\
            You are in gridworld. Make a move to help you reach the goal. 
            Your response must be some kind of move, even if you have to guess. 
            """)

    def execute(self, context=None):
        # move_response = self.llm.invoke(self.move_prompt)
        move_response = self.llm.invoke(self.move_prompt).content
        print("Move Response:", move_response)
        self.parent.message = move_response
        self.parent.update_context('feedback', move_response)


class ConvertStrategyToAction(Component):
    def __init__(self, llm, parent):
        self.llm = llm
        self.parent = parent
        self.convert_prompt_template = dedent("""\
            There are only 2 types of moves you can make:
            
            1. move forward
            2. turn left
            
            Come up with a combination of those two moves in order
            to successfully carry out the action: {strategy}
            
            Your final answer should be in the format of a python list 
            of moves, where each move is one of the 2 types listed above.
            E.g. ['move forward', 'turn left']
            """)

    def execute(self, context=None):
        outer_message = self.parent.message
        prompt = self.convert_prompt_template.format(strategy=outer_message)
        print("Response from convert_strategy:\n", prompt)
        
        pydantic_parser = PydanticOutputParser(pydantic_object=ListOfMoves)
        format_instructions = pydantic_parser.get_format_instructions()
        prompt_with_instructions = f"{prompt}\n\nFormat instructions: {format_instructions}"
        
        # response = self.llm.invoke(prompt_with_instructions)
        response = self.llm.invoke(prompt_with_instructions).content
        try:
            parsed_response = pydantic_parser.parse(response)
            print("Parsed Moves:", parsed_response.moves)
            self.parent.action_choices.append(parsed_response.moves)
            self.parent.update_context('action_choices', parsed_response.moves)
        except OutputParserException as e:
            print("Failed to parse response:", e)

############################################################################

if __name__=="__main__": 

    load_dotenv("/home/bhsu/.env") 
    # os.environ["GOOGLE_API_KEY"]=api_key
    llm=ChatGoogleGenerativeAI(model="gemini-pro")
    lact_chain=LactChainA(llm)
    action=lact_chain.propose_action()
    breakpoint()