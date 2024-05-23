import sys, os
sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain')
from classes.environment import AbstractEnvironment
from classes.learning import LearningScheme
from classes.reward import AbstractRewardFunction
from classes.lactchain import LactChain, Component, Context
from classes.state import State, InputDict
from ARGO_WRAPPER.ArgoLLM import ArgoLLM
from ARGO_WRAPPER.CustomLLM import ARGO_EMBEDDING
from ARGO_WRAPPER.ARGO import ArgoWrapper, ArgoEmbeddingWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
import pandas as pd


class SimpleGridRewardFunction(AbstractRewardFunction):
    def __init__(self, goal_position=(2, 2)):
        self.goal_position = goal_position

    def compute_reward(self, next_state):
        # Check if the goal is reached
        if next_state.attributes['x'] == 2 and next_state.attributes['y'] == 2:  # Assuming goal is at (2,2)
            return 100  # Reward for reaching the goal
        else:
            return -1  # Penalize each move to encourage shortest path

class SimpleEnvironment(AbstractEnvironment):
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        # Initialize with State instances using InputDict for positions
        self.current_state = State(attributes=InputDict({'x': 0, 'y': 0, 'orientation': 0}))
        self.lact_chains = []
        self.reward_function = SimpleGridRewardFunction((self.grid_size-1, self.grid_size-1))  # Initialize the reward function

    def reset(self):
        self.current_state.attributes['x'] = 0  # Reset position within the existing State instance
        self.current_state.attributes['y'] = 0
        return self.current_state
    
    def goal_criteria(self):
        # Goal is reached if the agent is at the bottom-right corner of the grid
        x, y = self.current_state.attributes['x'], self.current_state.attributes['y']
        return x == self.grid_size - 1 and y == self.grid_size - 1

    def step(self, lact_chain):
        # Get the current orientation and position
        current_orientation = self.current_state.attributes['orientation']
        x, y = self.current_state.attributes['x'], self.current_state.attributes['y']
        context = HistoryContext()

        # Determine the action based on the lact_chain
        action = lact_chain.execute(context)
        #print("Action:", context.get()[0])

        new_x = x
        new_y = y

        # Define movement based on orientation
        if action.get()[0] == 'move forward':
            if current_orientation == 0:  # facing up
                new_y = y - 1
            elif current_orientation == 1:  # facing right
                new_x = x + 1
            elif current_orientation == 2:  # facing down
                new_y = y + 1
            elif current_orientation == 3:  # facing left
                new_x = x - 1
        elif action.get()[0] == 'turn left':
            new_orientation = (current_orientation - 1) % 4
            self.current_state.attributes['orientation'] = new_orientation
            new_x, new_y = x, y  # No movement, just turn

        # Enforce boundary conditions
        new_x = max(0, min(new_x, self.grid_size - 1))
        new_y = max(0, min(new_y, self.grid_size - 1))

        # Update the state with the new valid position
        self.current_state.attributes['x'] = new_x
        self.current_state.attributes['y'] = new_y

        # Use the reward function to compute the reward
        reward = self.reward_function.compute_reward(self.current_state)
        done = self.goal_criteria()  # Check if goal is reached based on next_state

        return self.current_state, reward, done

    def get_current_state(self):
        return self.current_state  # Return the reference to the initial state
    
    def get_position(self):
        return self.current_state.attributes['x'], self.current_state.attributes['y']
    
    def close(self):
    # Add any necessary cleanup code here
        pass

class LactChainA(LactChain):
    def __init__(self):
        super().__init__()
        self.add_component(ComponentA())

class LactChainB(LactChain):
    def __init__(self):
        super().__init__()
        self.add_component(ComponentB())

class HistoryContext(Context):
    def __init__(self):
        super().__init__()
        self.history = []

    def update(self, key, value):
        # Update the context with a key-value pair
        super().update(key, value)
        # Append the action to the history list if it's an action update
        if key == 'action':
            self.history.append(value)

    def get(self):
        # Return the history of actions
        return self.history

class ComponentA(Component):
    def execute(self, context):
        # Logic for "Move forward"
        #print("Moving forward")
        context.update('action', 'move forward')

class ComponentB(Component):
    def execute(self, context):
        # Logic for "Turn left"
        #print("Turning left")
        context.update('action', 'turn left')


if __name__=="__main__": 

    env=SimpleEnvironment()

    observation, info = env.reset()

    print('DONE')