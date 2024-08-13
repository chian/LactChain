"""Interface for all language model generators to follow."""

from __future__ import annotations

from typing import Protocol, Any
from abc import ABC, abstractmethod
from lactchain.configs.base_config import BaseConfig


class LLMGenerator(ABC):
    """Generator protocol for all generators to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the generator with the configuration."""
        ...

    @abstractmethod
    def generate(self, prompts: str | list[str]) -> list[str]:
        """Generate response text from prompts.

        list[str]
            A list of responses generated from the prompts
            (one response per prompt).
        """
        ...
        
    @abstractmethod
    def tokenize(self, prompts: str | list[str]) -> list[dict[str, Any]]: 
        '''Tokenize a list of prompts and returns inputs
        
        list[Dict[str, Any]]
            A list of inputs output by tokenizer
        '''
        ...