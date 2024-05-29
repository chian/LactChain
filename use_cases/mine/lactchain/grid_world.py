import sys, os
from ollama import Client
from langchain_community.llms import Ollama

sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')

from use_cases.testcase.GridWorld_LLM import (LlamaClientWrapper, ValueFunction, ValueFunctionLearningScheme, 
                                              QLearningDataset, QLearning, SimpleGridRewardFunction, 
                                              SimpleEnvironment, State, LactChainA)
from classes.environment import AbstractEnvironment
from classes.state import State, InputDict

class MySimpleEnvironment(AbstractEnvironment):
    def __init__(self, grid_size=4, context=None):
        self.grid_size = grid_size
        # Initialize with State instances using InputDict for positions
        self.current_state = State(attributes=InputDict({'x': 0, 'y': 0, 'orientation': 0}))
        self.reward_function = SimpleGridRewardFunction((self.grid_size-1, self.grid_size-1))  # Initialize the reward function

    def reset(self):
        self.current_state.attributes['x'] = 0  # Reset position within the existing State instance
        self.current_state.attributes['y'] = 0
        return {'x': self.current_state.attributes['x'], 
                'y': self.current_state.attributes['y'], 
                'orientation':self.current_state.attributes['orientation']}
    
    def goal_criteria(self):
        # Goal is reached if the agent is at the bottom-right corner of the grid
        x, y = self.current_state.attributes['x'], self.current_state.attributes['y']
        return x == self.grid_size - 1 and y == self.grid_size - 1

    def step(self, action_choice):
        # Get the current orientation and position
        current_orientation = self.current_state.attributes['orientation']
        x, y = self.current_state.attributes['x'], self.current_state.attributes['y']

        new_x = x
        new_y = y

        # Iterate through each action in the action_choice list
        for action in action_choice:
            action_type = action  # Assuming action.get() returns a list with the action type as the first element

            # Define movement based on orientation and action type
            if action_type == 'move forward':
                if current_orientation == 0:  # facing up
                    new_y -= 1
                elif current_orientation == 1:  # facing right
                    new_x += 1
                elif current_orientation == 2:  # facing down
                    new_y += 1
                elif current_orientation == 3:  # facing left
                    new_x -= 1
            elif action_type == 'turn left':
                current_orientation = (current_orientation - 1) % 4
                self.current_state.attributes['orientation'] = current_orientation
            # Enforce boundary conditions after each action
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


if __name__=="__main__": 

    llm = Ollama(model="llama3")
    env = SimpleEnvironment(grid_size=3)
    env.lact_chains = [LactChainA(llm=llm)]
    val_func = ValueFunction(model_name="mistralai/Mistral-7B-Instruct-v0.2")

    breakpoint()