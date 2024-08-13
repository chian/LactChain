from __future__ import annotations

'''Class that parses and processes outputs from a RL environment'''
import numpy as np
import torch

from torch import Tensor
from typing import Literal, Optional
from pydantic import Field
from collections import deque
import random
from torch.utils.data import Dataset

from lactchain.configs import BaseConfig
from lactchain.utils import batch_data, join_inputs


class CriticBufferConfig(BaseConfig):
    '''Config for critic buffer'''

    name: Literal['buffer'] = 'buffer'

    max_buffer_size: int = Field(
        default=10000000,
        description='maximum buffer size per buffer'
    )

    dataset_size: int = Field(
        default=10000,
        description='Dataset size required to write to huggingface dataset'
    )

    batch_size: int = Field(
        default=128,
        description='The batch size to add data to buffer as'
    )


class CriticBuffer:
    '''Class that stores data for the writer to write to dataset'''

    def __init__(self, config: CriticBufferConfig) -> None:

        rewards = deque(maxlen=config.max_buffer_size)
        observations = deque(maxlen=config.max_buffer_size)
        infos = deque(maxlen=config.max_buffer_size)

        # pin these as attributes
        self.rewards = rewards
        self.observations = observations
        self.infos = infos

        # sample related attributes
        self.max_buffer_size = config.max_buffer_size
        self.dataset_size = config.dataset_size
        self.batch_size = config.batch_size

    def add_batch(self,
                  rewards: np.ndarray | Tensor,
                  observations: list[str],
                  infos: Optional[list[str]],
                  ) -> None:
        '''Takes in batches of data from environment and adds them to datastruct'''

        self.rewards.extend(rewards)
        self.observations.extend(observations)
        if infos:
            self.infos.extend(infos)

    def __iter__(self):
        '''Yields batches of data from built up buffer'''
        
        buffer_size = len(self.rewards)

        assert buffer_size == self.max_buffer_size, \
            f'''Buffer size not big enough for dataset, keep sampling'''

        indices = list(range(buffer_size))
        random.shuffle(indices)

        for start_idx in range(0, buffer_size, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            yield {
                'rewards': [self.rewards[i] for i in batch_indices],
                'observations': [self.observations[i] for i in batch_indices],
                'infos': [self.infos[i] for i in batch_indices]
            }

    def clear(self) -> None:
        '''Resets Buffer'''

        self.rewards = deque(maxlen=self.max_buffer_size)
        self.observations = deque(maxlen=self.max_buffer_size)
        self.infos = deque(maxlen=self.max_buffer_size)
        self.prompts = deque(maxlen=self.max_buffer_size)
        self.inputs = {}
        
    def __len__(self) -> int: 
        return len(self.rewards)
