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
import matplotlib.pyplot as plt
import pandas as pd

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x
    
class PolicyNetworkLearningScheme(LearningScheme):
    def __init__(self, policy_network, learning_rate=0.01):
        self.policy_network = policy_network
        self.optimizer = Adam(self.policy_network.parameters(), lr=learning_rate)

    def update_model(self, state, action, reward, next_state, done):
        state = torch.tensor([state.attributes['x'], state.attributes['y'], state.attributes['orientation']], dtype=torch.float32)
        next_state = torch.tensor([next_state.attributes['x'], next_state.attributes['y'], next_state.attributes['orientation']], dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.int64)
        reward = torch.tensor([reward], dtype=torch.float32)

        # Get current Q values
        current_q_values = self.policy_network(state)
        if current_q_values.ndim == 1:
            current_q_values = current_q_values.unsqueeze(0)  # Add a batch dimension if necessary
        current_q_values = current_q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)

         # Compute the expected Q values
        next_q_values = self.policy_network(next_state)
        if next_q_values.ndim == 1:
            next_q_values = next_q_values.unsqueeze(0)  # Add a batch dimension if necessary
        next_q_values = next_q_values.max(1)[0]
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

class QLearningAgent(AbstractRLAgent):
    def __init__(self, input_dim, output_dim, learning_rate=0.005, epsilon=1.0, epsilon_decay=0.999999, min_epsilon=0.1):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.policy_network = PolicyNetwork(input_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.learning_scheme = PolicyNetworkLearningScheme(self.policy_network)

    def select_action(self, state):
           if np.random.rand() < self.epsilon:
               action = np.random.choice(self.policy_network.output_dim)
           else:
               position = torch.tensor([state.attributes['x'], state.attributes['y'], state.attributes['orientation']], dtype=torch.float32)
               with torch.no_grad():
                   action_probs = self.policy_network(position)
               action = torch.argmax(action_probs).item()
           return action
    
    def learn(self, state, action, reward, next_state, done):
        self.learning_scheme.update_model(state, action, reward, next_state, done)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

class SimpleEnvironment(AbstractEnvironment):
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        # Initialize with State instances using InputDict for positions
        self.current_state = State(attributes=InputDict({'x': 0, 'y': 0, 'orientation': 0}))
        self.lact_chains = []

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

        # Proposed new position based on the action
        new_x = self.current_state.attributes['x']
        new_y = self.current_state.attributes['y']

        if action == 0:  # up
            new_y -= 1
        elif action == 1:  # right
            new_x += 1
        elif action == 2:  # down
            new_y += 1
        elif action == 3:  # left
            new_x -= 1

        # Enforce boundary conditions
        new_x = max(0, min(new_x, self.grid_size - 1))
        new_y = max(0, min(new_y, self.grid_size - 1))

        # Update the state with the new valid position
        self.current_state.attributes['x'] = new_x
        self.current_state.attributes['y'] = new_y

        reward = -1  # Penalize each move to encourage shortest path
        done = self.goal_criteria()  # Check if goal is reached
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

def train(agent, env, num_episodes=1000):
    """
    Train a reinforcement learning agent in a given environment.

    Args:
        agent: The agent to be trained, which must have select_action and learn methods.
        env: The environment in which the agent operates, must have reset and step methods.
        num_episodes: The number of episodes to train the agent for.
    """
    max_steps_per_episode = 20
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, done = env.step(env.lact_chains[action])
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            #print(f"Step {steps + 1}: Action taken = {action}: Position = {state.attributes['x'], state.attributes['y']}")
            steps += 1
        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Calculate rolling average of the rewards
    window_size = 50  # Define the size of the window for the rolling average
    # Convert the list of rewards to a pandas Series
    print("Length of rewards:",len(rewards))
    rolling_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    # Correct the x-axis values for the rolling average plot
    x_values = np.arange(window_size - 1, num_episodes)

    print("Length of Rolling Average:",len(rolling_avg),"\nLength of X Values:",len(x_values))
    #print(rolling_avg)

    # Plotting the rewards and the rolling average
    plt.figure(figsize=(10, 5))
    #plt.plot(rewards, label='Total Reward per Episode', alpha=0.5)  # Slightly transparent
    plt.plot(x_values, rolling_avg, label='Rolling Average', color='red')  # Rolling average line
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards Over Time During Training')
    plt.legend()
    plt.grid(True)
    plt.show()
    return rewards


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
    env.close()

    #Test of LearningScheme on Gridworld
    env = SimpleEnvironment(grid_size=3)
    env.lact_chains = [LactChainA(), LactChainB()]
    action_space_size = len(env.lact_chains)
    agent = QLearningAgent(input_dim=3, output_dim=action_space_size)
    print("Action Space:", action_space_size)
    rewards = train(agent, env, num_episodes=100000)
    
