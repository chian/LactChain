from __future__ import annotations

'''Class that saves tokens and embeddings, and others to a dataset and merges it'''

from typing import Optional, Literal
from pathlib import Path
from datasets import Dataset as HFDataset, concatenate_datasets
from tqdm import tqdm

from lactchain.configs import BaseConfig
from lactchain.generators import generator_factory
from lactchain.critic.embedder import EmbedderFunctionConfig, EmbedderFunction
from lactchain.critic.tokenizer import CriticTokenizerConfig, CriticTokenizer

class HuggingFaceWriterConfig(BaseConfig):
    """Configuration for the hugging face writer."""

    name: Literal['huggingface'] = 'huggingface'  # type: ignore[assignment]

    # The number of processes to use for writing the dataset
    num_proc: Optional[int] = None  # noqa: UP007

class Writer: 
    '''Class for writing the tokens, embeddings, and strings into a huggingface dataset
    
    Dataset structure for Writer: 
    
    rewards: Tensor
    
    observations: str
    
    infos: str
    
    inputs: dict[str, Any]
    
    embeddings: Tensor
    
    '''
    def __init__(self, num_proc: Optional[int]=None) -> None:
        """Initialize the writer with the configuration."""
        self.num_proc = num_proc

    def write(
        self,
        output_dir: Path,
        rewards: list[str],
        observations: list[str],
        infos: list[str]
    ) -> None:
        """Write the embeddings to disk.

        Parameters
        ----------
        output_dir : Path
            The output directory to write the dataset to.
        paths : list[str]
            The paths for the dataset.
        text : list[str]
            The text for the dataset.
        responses : list[str]
            The responses for the dataset.
        """
        # Create a dataset
        dataset = HFDataset.from_dict(
            mapping={
                'rewards':rewards,
                'observations': observations,
                'infos': infos,
            },
        )

        # Write the dataset to disk
        dataset.save_to_disk(output_dir)

    def merge(self, dataset_dirs: list[Path], output_dir: Path) -> None:
        """Merge the datasets from multiple directories.

        Parameters
        ----------
        dataset_dirs : list[Path]
            The dataset directories to merge.
        output_dir : Path
            The output directory to write the merged dataset to.
        """
        # Load all the datasets
        all_datasets = []
        for p in tqdm(dataset_dirs):
            # TODO: Debug why for some datasets, we have missing data
            try:
                dataset = HFDataset.load_from_disk(p)
            except FileNotFoundError:
                print(f'Skipping dataset {p} as it is missing.')
                continue
            all_datasets.append(dataset)

        # Concatenate the datasets
        dataset = concatenate_datasets(all_datasets)

        # Write the dataset to disk
        dataset.save_to_disk(output_dir, num_proc=self.num_proc)
