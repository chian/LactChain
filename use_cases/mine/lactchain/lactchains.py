import sys, os
from langchain_community.llms import Ollama
from textwrap import dedent
from typing import Any, List, Union, Dict
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')
from classes.lactchain import LactChain, Context, Component


MOVE_PROMPT=dedent("""\
            You are in gridworld. Make a move to help you reach the goal. 
            Your response must be some kind of move, even if you have to guess. 
            """)

CONVERSION_TEMPLATE=dedent("""\
            There are only 2 types of moves you can make:
    
            1. move forward
            2. turn left
    
        Come up with a combination of those two moves in order
        to successfully carry out the action: 
        {strategy}
    
        Your final answer should be in the format of a python list 
        of moves, where each move is one of the 2 types listed above.
        E.g. ['move forward', 'turn left']
        """)

# class ListOfMoves(BaseModel):
#     '''Pydantic class for storing moves as a list'''
#     moves: List[str]

# class LactChainA(LactChain):
#     def __init__(self, llm):
#         super().__init__()
#         self.llm = llm
#         self.message = ""
#         self.action_choices = []

#         self.strategy_component = self.create_strategy_component()
#         self.add_component(self.strategy_component)

#         self.converted_strategy_component = self.create_convert_strategy_component()
#         self.add_component(self.converted_strategy_component)

#         self.context=Context()

#     def add_feedback(self, message):
#         self.context.update('feedback', message)

#     def add_action(self, action):
#         self.context.update('action_choices', action)

#     def create_strategy_component(self):
#         return DeclareStrategyComponent(self.llm, self)

#     def create_convert_strategy_component(self):
#         return ConvertStrategyToActionComponent(self.llm, self)

#     def propose_action(self):
#         strategy_component = self.create_strategy_component()
#         strategy_component.execute()

#         convert_component = self.create_convert_strategy_component()
#         convert_component.execute()

#         return self.action_choices
    
# class DeclareStrategyComponent(Component):
#     def __init__(self, llm, parent):
#         self.llm = llm
#         self.parent = parent
#         self.move_prompt = dedent("""\
#             You are in gridworld. Make a move to help you reach the goal. 
#             Your response must be some kind of move, even if you have to guess.
#         """)

#     def execute(self, context=None):
#         move_response = self.llm.invoke(self.move_prompt)
#         print("Move Response:", move_response)
#         self.parent.message = move_response


# class ConvertStrategyToActionComponent(Component):
#     class ListOfMoves(BaseModel):
#         moves: List[str]

#     def __init__(self, llm, parent):
#         self.llm = llm
#         self.parent = parent
#         self.convert_prompt_template = dedent("""\
#             There are only 2 types of moves you can make:
#             1. move forward
#             2. turn left
#             Come up with a combination of those two moves in order to successfully carry out the action: {strategy}
#             Your final answer should be in the format of a python list of moves, where each move is one of the 2 types listed above.
#             E.g. ['move forward', 'turn left']
#         """)

#     def execute(self, context=None):
#         prompt = self.convert_prompt_template.format(strategy=self.parent.message)
#         print("Response from convert_strategy:", prompt)

#         pydantic_parser = PydanticOutputParser(pydantic_object=self.ListOfMoves)
#         format_instructions = pydantic_parser.get_format_instructions()
#         prompt_with_instructions = f"{prompt}\n\nFormat instructions: {format_instructions}"

#         response = self.llm.invoke(prompt_with_instructions)
#         try:
#             parsed_response = pydantic_parser.parse(response)
#             print("Parsed Moves:", parsed_response.moves)
#             self.parent.action_choices.append(parsed_response.moves)
#         except OutputParserException as e:
#             print("Failed to parse response:", e)


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
        move_response = self.llm.invoke(self.move_prompt)
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
        print("Response from convert_strategy:", prompt)
        
        pydantic_parser = PydanticOutputParser(pydantic_object=ListOfMoves)
        format_instructions = pydantic_parser.get_format_instructions()
        prompt_with_instructions = f"{prompt}\n\nFormat instructions: {format_instructions}"
        
        response = self.llm.invoke(prompt_with_instructions)
        
        try:
            parsed_response = pydantic_parser.parse(response)
            print("Parsed Moves:", parsed_response.moves)
            self.parent.action_choices.append(parsed_response.moves)
            self.parent.update_context('action_choices', parsed_response.moves)
        except OutputParserException as e:
            print("Failed to parse response:", e)


class ListOfMoves(BaseModel):
    moves: List[str]


############################################################################

if __name__=="__main__": 

    model_id = "microsoft/Phi-3-mini-128k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', trust_remote_code=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    hf = HuggingFacePipeline(pipeline=pipe)

    breakpoint()

    llm=Ollama(model="llama3")
    lact_chain=LactChainA(llm)
    action=lact_chain.propose_action()
    breakpoint()







# class DeclareStrategy(Component):
#     def __init__(self, 
#                  llm:Ollama, # langchain class 
#                  move_prompt:str
#                  ):
#         self.llm = llm
#         self.move_prompt = move_prompt
        
#     def execute(self, context:str=None):
#         # Retrieve the last feedback from context, or use a default value if none is available
#         # This didn't work very well - feedback mechanism needs to be rethought
#         strategy = self.llm.invoke(self.move_prompt)
#         return strategy

# class ConvertStrategytoAction(Component):
#     def __init__(self, 
#                  llm:Ollama, 
#                  conversion_prompt_template:str, 
#                  strategy:str, 
#                  parent:Any
#                  ):
#         self.llm = llm
#         self.convert_prompt_template = conversion_prompt_template
#         self.prompt = self.convert_prompt_template.format(strategy=strategy)
#         self.parent = parent
#         self.pydantic_parser=PydanticOutputParser(pydantic_object=ListOfMoves)

#     def execute(self, 
#                 context:str=None
#                 ):
#         print("Response from convert_strategy:", self.prompt)
#         format_instructions = self.pydantic_parser.get_format_instructions()
#         # input is prompt with instructions
#         response = self.llm.invoke(dedent(f'''{self.prompt}\n 
#                                           Format instructions: {format_instructions}''')) # LLM responds to prompt with instructions on how to move
#         try:
#             # Parse the LLM response into the Pydantic model
#             parsed_response = self.pydantic_parser.parse(response)
#             print("Parsed Moves:", parsed_response.moves)
#             # Add the parsed moves to the action_choices list in the outer class
#             self.parent.action_choices.append(parsed_response.moves)
#         except OutputParserException as e:
#             print("Failed to parse response:", e)

# class LactChainA(LactChain): 
#     def __init__(self, llm:Ollama, move_prompt:str): 
#         super().__init__()
#         self.llm=llm
#         self.move_prompt=move_prompt

#         self.strategy=self.create_strategy(self.llm, self.move_prompt)
#         self.converted_strategy=self.convert_strategy(self.llm)

#         self.strategy_component = self.create_strategy(llm)
#         self.add_component(self.strategy_component)
#         self.message = ""
#         self.converted_strategy_component = self.convert_strategy(llm)
#         self.add_component(self.converted_strategy_component)
#         self.action_choices = []

#     def add_feedback(self,message):
#         context.update('feedback', message)

#     def add_action(self, action):
#         context.update('action_choices', action)

#     def create_strategy(self, llm:Ollama, move_prompt:str): 
#         strategy=DeclareStrategy(llm, move_prompt)
#         return strategy
    
#     def convert_strategy(self, 
#                          llm:Ollama, 
#                          conversion_prompt_template:str, 
#                          strategy:str
#                          ):
#         outer_message=self.message
#         _strategy_converter=ConvertStrategytoAction(llm, conversion_prompt_template, strategy)
#         converted_strategy=_strategy_converter.execute()
#         return converted_strategy
    
#     def propose_action(self): 
        
#         ...


# class LactChainA(LactChain):
#     def __init__(self,llm):
#         super().__init__()
#         self.strategy_component = self.create_strategy(llm)
#         self.add_component(self.strategy_component)
#         self.message = ""
#         self.converted_strategy_component = self.convert_strategy(llm)
#         self.add_component(self.converted_strategy_component)
#         self.action_choices = []

#     def add_feedback(self,message):
#         context.update('feedback', message)

#     def add_action(self, action):
#         context.update('action_choices', action)

#     def create_strategy(self,llm):
    
#         class declare_strategy(Component):
#             def __init__(self,llm):
#                 self.llm = llm
#                 self.move_prompt = dedent("""\
#                     You are in gridworld. Make a move to help you reach the goal. 
#                     Your response must be some kind of move, even if you have to guess. 
#                     """)
#             def execute(self,context=None):
#                 # Retrieve the last feedback from context, or use a default value if none is available
#                 # This didn't work very well - feedback mechanism needs to be rethought
        
#                 move_response = self.llm.invoke(self.move_prompt)
#                 print("Move Response:", move_response)
#                 self.message = move_response

#         return declare_strategy(llm)

#     def convert_strategy(self,llm):
#         outer_message = self.message

#         class ListOfMoves(BaseModel):
#             moves: List[str]

#         class convert_strategy_to_action(Component):
#             def __init__(self,llm,parent):
#                 self.llm = llm
#                 self.parent = parent
#                 self.convert_prompt_template = dedent("""\
#                     There are only 2 types of moves you can make:
            
#                     1. move forward
#                     2. turn left
            
#                 Come up with a combination of those two moves in order
#                 to successfully carry out the action: {strategy}
            
#                 Your final answer should be in the format of a python list 
#                 of moves, where each move is one of the 2 types listed above.
#                 E.g. ['move forward', 'turn left']
#                 """)
#                 # Fill in the placeholder in the prompt template
#                 self.prompt = self.convert_prompt_template.format(strategy=outer_message)

#             def execute(self,context=None):
#                 print("Response from convert_strategy:", self.prompt)
#                 pydantic_parser = PydanticOutputParser(pydantic_object=ListOfMoves)
#                 format_instructions = pydantic_parser.get_format_instructions()

#                 #print("Prompt:", prompt)
#                 prompt_with_instructions = f"{self.prompt}\n\nFormat instructions: {format_instructions}"

#                 # LLM responds to prompt with instructions on how to move
#                 response = self.llm.invoke(prompt_with_instructions)
#                 #print("Response:", response)

#                 try:
#                     # Parse the LLM response into the Pydantic model
#                     parsed_response = pydantic_parser.parse(response)
#                     print("Parsed Moves:", parsed_response.moves)
#                     # Add the parsed moves to the action_choices list in the outer class
#                     self.parent.action_choices.append(parsed_response.moves)
#                 except OutputParserException as e:
#                     print("Failed to parse response:", e)

#         return convert_strategy_to_action(llm,self)
    
#     def propose_action(self):
#         #propose_action should create a new action_choices list entry by calling create_strategy and convert_strategy
#         self.create_strategy(llm)
#         self.convert_strategy(llm)
#         return self.action_choices #[-1] removed for testing