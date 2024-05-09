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

from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from textwrap import dedent

class Phi3LLM:
    def __init__(self):
        self.model_name = "microsoft/Phi-3-mini-128k-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="mps",  # Assuming CUDA is available
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def invoke(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        generation_args = {
            "max_new_tokens": 256,  # Adjust token limit as needed
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False
        }
        output = self.pipe(messages, **generation_args)
        return output[0]['generated_text']

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
    def __init__(self, grid_size=4, context=None):
        self.grid_size = grid_size
        # Initialize with State instances using InputDict for positions
        self.current_state = State(attributes=InputDict({'x': 0, 'y': 0, 'orientation': 0}))
        self.reward_function = SimpleGridRewardFunction((self.grid_size-1, self.grid_size-1))  # Initialize the reward function

    def reset(self):
        self.current_state.attributes['x'] = 0  # Reset position within the existing State instance
        self.current_state.attributes['y'] = 0
        return self.current_state
    
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
            action_type = action.get()[0]  # Assuming action.get() returns a list with the action type as the first element

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
                action.add_feedback('that worked!')
            elif action_type == 'turn left':
                current_orientation = (current_orientation - 1) % 4
                self.current_state.attributes['orientation'] = current_orientation
                # No movement, just turn
                action.add_feedback('that worked!')
            else:
                action.add_feedback('that did not work!')

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

class LactChainA(LactChain):
    def __init__(self,llm, context):
        super().__init__()
        self.add_component(new_action(llm))
    def add_feedback(self,message):
        context.update('feedback', message)
    def add_action(self, action):
        context.update('action_choices', action)

class HistoryContext(Context):
    def __init__(self):
        super().__init__()
        self.feedback = []
        self.action_choices = []
    def update(self, key, value):
        # Update the context with a key-value pair
        super().update(key, value)
        # Append the action to the history list if it's an action update
        if key == 'feedback':
            self.feedback.append(value)
        if key == 'action_choices':
            self.action_choices.append(value)
    def get(self, key):
        # Return the history based on the key
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f"No such key '{key}' in context.")

class new_action(Component):
    def __init__(self,llm):
        self.llm = llm
        self.move_prompt = dedent("""\
            You are in gridworld. Make a move. Your response should be short and directed. 
            Answer: """)
        self.convert_prompt_template = dedent("""\
            There are only 2 types of moves you can make:
            
                1. move forward
                2. turn left
            
            Come up with a combination of those two moves in order
            to successfully carry out the action: {action}
            
            Your final answer should be in the format of a python list 
            of moves, where each move is one of the 2 types listed above.
            E.g. ['move forward', 'turn left']
            Answer: """)
    def execute(self, context):
        # Retrieve the last feedback from context, or use a default value if none is available
        # This didn't work very well - feedback mechanism needs to be rethought
        #past_feedback = context.get() if context.get() else "no feedback, first move"
        
        move_response = self.llm.invoke(self.move_prompt)
        print("Move Response:", move_response)

        # Fill in the placeholder in the prompt template
        prompt = self.convert_prompt_template.format(action=move_response)
        print("Prompt:", prompt)

        # LLM responds to prompt with instructions on how to move
        response = self.llm.invoke(prompt)
        print("Response from new_action:", response)

        # To do - make this agenticChunking - add action to list of possible actions
        # Policy can choose old actions or to make new actions
        context.update('action', response)
        
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
    context = HistoryContext()
    context.update('action_choices', [LactChainA()])

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, done = env.step(context.get('action_choices')[action])
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
    # Example usage
    #torch.set_default_device("mps")
    phi3_llm = Phi3LLM()
    prompt = "Translate the following English text to French: | Hello, how are you?"
    response = phi3_llm.invoke(prompt)
    print("Response from Phi-3 LLM:", response)

    #Test of LactChain class
    context = HistoryContext()
    lact_chain = LactChainA(llm=phi3_llm,context=context)
    for i in range(1):
        context = lact_chain.execute(context)
        print("Context - action_choices:",context.get('action_choices'))

    exit(0)

    #Test of State
    attributes = InputDict({'x': 0, 'y': 0})
    state = State(attributes=attributes, textblock="some text")
    print("State:", state.textblock)
    print("State Dict:", state.attributes)

    #Test of Environment class
    env = SimpleEnvironment(grid_size=5,context=context)
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
    rewards = train(agent, env, num_episodes=1000000)
    
