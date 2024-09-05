from __future__ import annotations

'''Imports for environments'''

import gymnasium as gym
from typing import Dict

from lactchain.environments.grid_world import (GridWorldConfig, 
                                               VectorizedGridWorld, 
                                               GridEnvironment, 
                                               process_environment_outputs)

def make_env(env_id: str, config: Dict):
    '''Make GridWorld Environment based on config'''
    def _init():
        env = gym.make(env_id, **config)  # Pass config parameters to the environment
        return env
    return _init


def build_gridworld_env(config: GridWorldConfig):
    '''Make GridWorld Environment based on config'''
    
    def _init():
        env = VectorizedGridWorld(config)  # Pass config parameters to the environment
        return env

    return _init

