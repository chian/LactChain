"""Hugging face reader for reading text from disk."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Union
from torch import Tensor
from datasets import Dataset

from lactchain.configs import BaseConfig

Numerical = Union[
    int, 
    float, 
    Tensor, 
    
]

class HuggingFaceReaderConfig(BaseConfig):
    """Configuration for the hugging face reader."""

    name: Literal['huggingface'] = 'huggingface'  # type: ignore[assignment]


class HuggingFaceReader:
    """Hugging face reader for reading text from disk."""

    def __init__(self, config: HuggingFaceReaderConfig) -> None:
        """Initialize the reader with the configuration."""
        self.config = config

    def read(self, input_path: Path) -> tuple[list[str], list[str]]:
        """Read the dataset.

        Parameters
        ----------
        input_path : Path
            The path to the dataset.

        Returns
        -------
        tuple[list[str], list[str]]
            The text and paths from the dataset.
        """
        # Load the dataset from disk
        dataset = Dataset.load_from_disk(input_path)

        # Load the text and paths from the dataset
        rewards: list[int | float | Tensor] = dataset['rewards']
        observations: list[str] = 
        
        text: list[str] = dataset['text']
        paths: list[str] = dataset['path']
        return text, paths