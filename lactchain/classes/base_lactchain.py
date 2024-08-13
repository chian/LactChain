#This set of classes allows for heirarchical passing of language chains into execution chains

#Example:
# 2 components make up a SubChain in the below and then the MainChain can be composed of SubChains.

# class LLMCallComponent(Component):
#     def execute(self, context):
#         # Example LLM call modifying the context
#         context['state'] += " after LLM call"
#         context['llm_output'] = "example output"

# class ParseOutputComponent(Component):
#     def execute(self, context):
#         if 'llm_output' in context:
#             context['state'] += " parsed from " + context['llm_output']

# class SubChain(LactChain):
#     def __init__(self):
#         super().__init__()
#         self.add_component(LLMCallComponent())
#         self.add_component(ParseOutputComponent())

#     def execute(self, context):
#         super().execute(context)
#         context['state'] += " with final modifications in subchain"

# class MainChain(LactChain):
#     def __init__(self):
#         super().__init__()
#         self.add_component(SubChain())  # SubChain as a component
#         self.add_component(AnotherComponent())  # Another component that follows the subchain

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Optional

from lactchain.classes.base_generator import LLMGenerator
from lactchain.classes.base_prompt import BasePromptTemplate

class Context(ABC):
    """
    Abstract base class for context management in a LactChain.
    """
    def __init__(self):
        self.data = {}

    @abstractmethod
    def update(self, key, value):
        """
        Update the context with a key-value pair.
        """
        pass

    @abstractmethod
    def get(self, key, default=None):
        """
        Retrieve a value from the context by key, with an optional default.
        """
        pass

class Component(ABC):
    """
    Abstract base class for components in a LactChain.
    """
    @abstractmethod
    def execute(self, context=None):
        """
        Execute the component logic using the shared context.
        """
        pass

class LactChain(ABC):
    """
    Abstract base class for a language action chain (LactChain).
    """
    def __init__(self):
        self.components = []

    def add_component(self, component):
        """
        Add a component to the chain. Ensures that the component is an instance of Component.
        """
        if not isinstance(component, Component):
            raise TypeError("All components must be instances of Component")
        self.components.append(component)

    def execute(self, context=None):
        """
        Execute the chain of components on the given context.
        """
        for component in self.components:
            component.execute(context)
        return context

class ActorChain(ABC): 
    '''Abstract sub-chain that is meant to be a component of Lactchain
    Plan: StrategyChain --> Parser/Mapper Chain --> Output (both classes part of Actor Chain)
    '''
    def __init__(self, 
                 generator:LLMGenerator,
                 prompt_template:BasePromptTemplate, 
                 **kwargs
                 ) -> None:
        
        self.generator = generator
        self.prompt_template = prompt_template
    
    @abstractmethod    
    def _preprocess(self, **kwargs) -> list[str]: 
        '''
        Function that uses the prompt template to preprocess str data into appropriate prompt template
        '''

    @abstractmethod
    def parse_outputs(self, **kwargs) -> list[str]: 
        '''
        Method that parses the output prompts from the language model
        '''

    
    @abstractmethod
    def map_actions(self, **kwargs) -> Tuple[list[Any], list[Any]]: 
        '''
        Method that takes the parsed_outputs and maps them to actions
        '''

    @abstractmethod
    def sample_actions(self, **kwargs) -> Tuple[list[str], str]:
        '''
        send the final prompt into the model, then get the output, runs it through parse_outputs, and then 
        runs it through map_actions to generate a series of actions
        '''

    
    
class StrategyChain(ABC): 
    '''Abstract sub-chain that is meant to be a component of Lactchain
    Plan: StrategyChain --> Parser/Mapper Chain --> Output (both classes part of Actor Chain)
    '''
    def __init__(self, 
                 generator:LLMGenerator,
                 prompt_template:BasePromptTemplate
                 ) -> None:
        
        self.generator = generator
        self.prompt_template = prompt_template
       
    @abstractmethod 
    def _preprocess() -> list[str]: 
        '''
        Function that uses the prompt template to preprocess str data into appropriate prompt template
        '''
    
    @abstractmethod
    def sample_strategies() -> Tuple[list[str], str]:
        '''
        send the final prompt into the model, then get the output, runs it through parse_outputs, and then 
        runs it through map_actions to generate a series of actions
        '''
    

        