from __future__ import annotations

'''Templates for compiling output prompts from the environment'''

from typing import Any, Optional
from textwrap import dedent

from lactchain.configs import BaseConfig
from lactchain.classes.base_prompt import BasePromptTemplate


class GridWorldPromptTemplateConfig(BaseConfig):
    """Configuration for the StrategyPromptTemplate"""
    name: Literal['environment'] = 'environment'  # type: ignore[assignment]
    
    

class GridWorldOutput(BasePromptTemplate): 
    """GridWorld output prompt template."""
    
    template_with_context: str = dedent('''{state}\n{info}''')

    template_no_context: str = dedent('''{state}''')
    
    def __init__(self, config: GridWorldPromptTemplateConfig) -> None:
        """Initialize the QuestionAnswerPromptTemplate."""
        self.config = config

    def _format_prompt(
        self,
        state: str, 
        info: Optional[str]
        ) -> str:
        """Format the prompt with the question and context."""
        
        if info is not None:     
            return self.template_with_context.format(
                state = state, 
                info = info
            )
        else: 
            return self.template_no_context.format(
                state = state
            )

    def preprocess(
        self,
        states: str | list[str], 
        infos: Optional[str | list[str]]
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
        if isinstance(states, str) or isinstance(infos, str):
            states = [states]
            infos = [infos]
            
        # If no contexts are provided, use the no-context template
        if infos is None:
            return list(map(self._format_prompt, states))

        # Build the prompts using the template
        return list(map(self._format_prompt, states, infos))

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