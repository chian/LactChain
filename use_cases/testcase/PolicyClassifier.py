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


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x
    
class SimpleEnvironment:
    def __init__(self):
        self.current_state = None

    def reset(self):
        self.current_state = self.get_initial_state()
        return self.current_state

    def step(self, action):
        # Logic to determine the next state and reward based on action
        next_state = None
        reward = None
        done = False
        return next_state, reward, done

    def get_initial_state(self):
        # Return an initial state
        return [0.0] * 10  # Example state vector

class QLearningAgent(AbstractRLAgent):
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        self.policy_network = PolicyNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_probs = self.policy_network(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def learn(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.int64)

        # Get current Q values
        current_q_values = self.policy_network(state)[action]

        # Compute the expected Q values
        next_q_values = self.policy_network(next_state).max()
        expected_q_values = reward + (0.99 * next_q_values * (1 - int(done)))

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(0))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class LactChainA(LactChain):
    def __init__(self):
        super().__init__()
        # Add components specific to LactChain A
        self.add_component(ComponentA())
        # Additional components can be added here

class LactChainB(LactChain):
    def __init__(self):
        super().__init__()
        # Add components specific to LactChain B
        self.add_component(ComponentB())
        self.add_component(ComponentA())
        # Additional components can be added here

class LactChainB(LactChain):
    def __init__(self):
        super().__init__()
        # Add components specific to LactChain B
        self.add_component(ComponentB())
        self.add_component(ComponentB())
        self.add_component(ComponentB())
        self.add_component(ComponentA())
        # Additional components can be added here

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
    context = HistoryContext()
    lact_chain = LactChainA()
    lact_chain.execute(context)
    print(context.get())
    #train()

