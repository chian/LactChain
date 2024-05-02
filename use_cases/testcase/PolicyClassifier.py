import os
print("Current Working Directory:", os.getcwd())
import sys
print("Python Path:", sys.path)
sys.path.append('/Users/chia/Documents/ANL/Software/LactChain/')

from classes.agent import AbstractRLAgent
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

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x
    
class PolicyNetworkLearningScheme(LearningScheme):
    def __init__(self, policy_network, learning_rate=0.01):
        self.policy_network = policy_network
        self.optimizer = Adam(self.policy_network.parameters(), lr=learning_rate)

    def update_model(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.int64)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        # Get current Q values
        current_q_values = self.policy_network(state).gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Compute the expected Q values
        next_q_values = self.policy_network(next_state).max(1)[0]
        expected_q_values = reward + (0.99 * next_q_values * (1 - done))

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, filepath):
        torch.save(self.policy_network.state_dict(), filepath)

    def load_model(self, filepath):
        self.policy_network.load_state_dict(torch.load(filepath))
    
class SimpleEnvironment(AbstractEnvironment):
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        # Initialize with State instances using InputDict for positions
        self.current_state = State(attributes=InputDict({'x': 0, 'y': 0, 'orientation': 0}))

    def reset(self):
        self.current_state.attributes['x'] = 0  # Reset position within the existing State instance
        self.current_state.attributes['y'] = 0
        return self.current_state
    
    def goal_criteria(self):
        # Goal is reached if the agent is at the bottom-right corner of the grid
        x, y = self.current_state.attributes['x'], self.current_state.attributes['y']
        return x == self.grid_size - 1 and y == self.grid_size - 1

    def step(self, lact_chain):
        # Define actions as 0: up, 1: right, 2: down, 3: left
        action = lact_chain.determine_action(self.current_state)

        # Update position within the existing State instance
        self.current_state.attributes['x'] = 0  # Reset position within the existing State instance
        self.current_state.attributes['y'] = 0

        reward = -1  # Penalize each move to encourage shortest path
        done = self.goal_criteria  # Check if goal is reached
        if done:
            reward = 100  # Reward for reaching the goal
        return self.current_state, reward, done

    def get_current_state(self):
        return self.current_state  # Return the reference to the initial state
    
    def get_position(self):
        return self.current_state.attributes['x'], self.current_state.attributes['y']
    
    def close(self):
    # Add any necessary cleanup code here
        pass

class QLearningAgent(AbstractRLAgent):
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        self.policy_network = PolicyNetwork(input_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.tensor(state.get_position(), dtype=torch.float32)  # Assuming State has a method to get position
        with torch.no_grad():
            action_probs = self.policy_network(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

class LactChainA(LactChain):
    def __init__(self):
        super().__init__()
        self.add_component(ComponentA())

    def determine_action(self, current_state):
        # Assuming current_state includes 'orientation' and 'x', 'y' coordinates
        orientation = current_state.attributes['orientation']
        # Mapping of orientation to action codes: 0: up, 1: right, 2: down, 3: left
        action_map = {0: 0, 1: 1, 2: 2, 3: 3}
        # Example: if orientation is 1 (facing right), "move forward" translates to moving right
        return action_map[orientation]

class LactChainB(LactChain):
    def __init__(self):
        super().__init__()
        self.add_component(ComponentB())
        self.add_component(ComponentA())

    def determine_action(self, current_state):
        orientation = current_state.attributes['orientation']
        # Let's assume this chain sometimes wants to turn left before moving forward
        # Rotate left (decrease orientation by 1, wrap around using % 4)
        new_orientation = (orientation - 1) % 4
        current_state.attributes['orientation'] = new_orientation
        action_map = {0: 0, 1: 1, 2: 2, 3: 3}
        return action_map[new_orientation]

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
        print("Moving forward")
        context.update('action', 'move forward')

class ComponentB(Component):
    def execute(self, context):
        # Logic for "Turn left"
        print("Turning left")
        context.update('action', 'turn left')

def train():
    env = SimpleEnvironment()
    agent = QLearningAgent(input_dim=10, output_dim=2)  # Assuming two actions

    for episode in range(1000):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state

if __name__ == "__main__":
    #Test of LactChain class
    context = HistoryContext()
    lact_chain = LactChainA()
    lact_chain.execute(context)
    print(context.get())

    #Test of State
    attributes = InputDict({'x': 0, 'y': 0})
    state = State(attributes=attributes, textblock="some text")
    print("State:", state.textblock)
    print("State Dict:", state.attributes)

    #Test of Environment class
    env = SimpleEnvironment(grid_size=5)
    initial_state = env.reset()
    print("Initial State:", initial_state)
    lact_chain_a = LactChainA()
    lact_chain_b = LactChainB()
    env.step(lact_chain_a)
    env.step(lact_chain_b)
    print("Final State after actions:", env.get_current_state().attributes)


    #train()

