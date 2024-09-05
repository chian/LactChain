from __future__ import annotations

'''Import buffers and datasets'''

from typing import Union

from lactchain.datasets.critic_buffer import CriticBuffer, CriticBufferConfig


Buffers=Union[
    CriticBuffer
]

BufferConfig=Union[
    CriticBufferConfig
]

BUFFERS={
    'action':..., 
    'critic': (CriticBuffer, CriticBufferConfig)
}


def get_buffer(config: BufferConfig) -> Buffers: 
    '''initializes buffer based on strategy'''
    
    kwargs = config.model_dump()
    name = kwargs.get('name')
    
    strategy = BUFFERS.get(name)
    if not strategy: 
        raise ValueError(
            f'Unknown generator name: {name}.'
            f' Available: {set(BUFFERS.keys())}',
        )
        
    cls, cfg = strategy
    
    return cls(cfg(**kwargs))