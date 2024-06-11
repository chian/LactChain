import os
print("Current Working Directory:", os.getcwd())
import sys
print("Python Path:", sys.path)
sys.path.append('/Users/chia/Documents/ANL/Software/LactChain/')

from classes.environment import AbstractEnvironment
from classes.learning import LearningScheme
from classes.reward import AbstractRewardFunction
from classes.lactchain import LactChain, Component, Context
from classes.state import State, InputDict


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
import pandas as pd

from textwrap import dedent

from ollama import Client
from langchain_community.llms import Ollama
from pydantic import BaseModel
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException

from peft import get_peft_model, LoraConfig, get_peft_config
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

class LlamaClientWrapper:
    def __init__(self):
        self.client = Client(host='http://localhost:11434')

    def invoke(self, prompt):
        response = self.client.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']

class FoundationModel(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        super(FoundationModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name,output_hidden_states=True)
        self.config = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.hidden_states[-1]
        return last_hidden_states

class ValueFunctionHead(nn.Module):
    def __init__(self, foundation_model, num_layers=1):
        super(ValueFunctionHead, self).__init__()
        self.base_model = foundation_model
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=foundation_model.config.hidden_size, nhead=8), num_layers=num_layers)
        self.q_value_head = nn.Linear(foundation_model.config.hidden_size, 1)

    def forward(self, inputs, attention_mask=None):
        # Pass the embedded state vector through the transformer encoder
        transformer_output = self.transformer(inputs)
        # Compute the Q-values using the linear head
        q_values = self.q_value_head(transformer_output)
        # Take the mean of the Q-values to get a single predicted Q-value
        predicted_q_value = q_values.mean()
        return predicted_q_value

class ValueFunctionLearningScheme(LearningScheme):
    def __init__(self, value_function_head):
        self.model = value_function_head
        self.trainer = None
        self.training_args = TrainingArguments(
            output_dir="./model_output",
            num_train_epochs=1,  # Set to 1 for episodic training
            per_device_train_batch_size=16,
            learning_rate=1e-4,  
            logging_dir='./logs',
            logging_steps=10,
        ) 
    class CustomTrainer(Trainer):
        def __init__(self, model, args, train_dataset, dtype=None):
            super().__init__(model, args, train_dataset)
            self.dtype = dtype
        def compute_loss(self,model, inputs, return_outputs=False):
            # Assuming inputs contain 'input_ids' and 'target_q_value'
            target_q_values = inputs.get("target_q_value").unsqueeze(-1)  # Ensure target_q_values are correctly shaped
            outputs = model(input_ids=inputs['input_ids'])  # Call model without attention_mask
            loss = torch.nn.functional.mse_loss(outputs, target_q_values)

            if return_outputs:
                return (loss, outputs)
            else:
                return loss
        
    def update_model(self, train_dataset):
        self.trainer = self.CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
        )
        print("Trainer:", self.trainer)
        self.trainer.train()
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))

class QLearningDataset(torch.utils.data.Dataset):
    def __init__(self, foundation_model, val_func, tokenizer, env, num_samples, gamma=0.99):
        self.base_model = foundation_model
        self.val_func = val_func
        self.tokenizer = tokenizer
        self.env = env
        self.gamma = gamma
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.target_q_values = []
        self.max_steps = 10
        #getting target and predicted q-values, respectively
        self.prompt_template = dedent("""\
        You are in a gridworld. The agent is at position {x}, {y} with 
        orientation {orientation}.
        """)
        self.sample_experience(num_samples)
        self.target_q_values = self.compute_target_q_value()
        self.predicted_q_values = self.get_predicted_q_values()
        self.predictions = self.predicted_q_values 
        self.inputs = self.get_inputs()
        self.labels = self.target_q_values

    def sample_experience(self, num_samples):
        for _ in range(num_samples):
            state = self.env.reset()  # Start a new episode
            done = False
            steps = 0
            while not done:
                action = self.env.lact_chains[0].propose_action()  # Get action from the environment's LactChain
                print("Action (sample_experience):", action)
                next_state, reward, done = self.env.step(action)  # Execute the action
                print("Next State:", next_state.attributes['x'], next_state.attributes['y'], next_state.attributes['orientation'])
                print("Reward:", reward)

                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.next_states.append(next_state)
                self.dones.append(done)
                print("state -> action -> next_state: (", 
                      state.attributes['x'], ",", state.attributes['y'], ",", 
                      state.attributes['orientation'], ") -> ", action, " -> (", 
                      next_state.attributes['x'], ",", next_state.attributes['y'], ",", 
                      next_state.attributes['orientation'], ")")
                # Compute target Q-value for this transition
                #self.target_q_values = self.compute_target_q_value()

                env.update_state()  # Move to the next state
                print("Done:", done)
                steps += 1
                if steps > self.max_steps:
                    done = True
        self.inputs = self.get_inputs()

    def compute_target_q_value(self):
        target_q_values = []
        for next_state, reward, done in zip(self.next_states, self.rewards, self.dones):
            if done:
                target_q_values.append(reward)
            else:
                self.base_model.eval()
                with torch.no_grad():
                    # Predict Q-values for the next state
                    next_state_prompt = self.prompt_template.format(x=next_state.attributes['x'], 
                                                                    y=next_state.attributes['y'], 
                                                                    orientation=next_state.attributes['orientation'])
                    next_state_inputs = self.tokenizer(next_state_prompt, return_tensors="pt")
                    valfunc_inputs = self.base_model(**next_state_inputs)
                    next_state_q_value = self.val_func(valfunc_inputs)
                    print("Next state q value:", next_state_q_value, len(self.states))
                    target_q_value = reward + self.gamma * next_state_q_value
                    target_q_values.append(target_q_value.item())
        self.base_model.train()
        self.target_q_values.extend(target_q_values)

    #Not actually used by trainer directly but part of computing loss
    def get_predicted_q_values(self):
        self.base_model.eval()
        with torch.no_grad():
            predicted_q_values = []
            for state in self.states:
                state_prompt = self.prompt_template.format(x=state.attributes['x'], 
                                                           y=state.attributes['y'], 
                                                           orientation=state.attributes['orientation'])
                state_inputs = self.tokenizer(state_prompt, return_tensors="pt")
                valfunc_inputs = self.base_model(**state_inputs)
                predicted_q_value = self.val_func(valfunc_inputs)
                predicted_q_values.append(predicted_q_value)
        self.base_model.train()
        return predicted_q_values
    
    def get_inputs(self):
        inputs = []
        for state in self.states:
            state_prompt = self.prompt_template.format(x=state.attributes['x'], 
                                                       y=state.attributes['y'], 
                                                       orientation=state.attributes['orientation'])
            input_ids = self.tokenizer.encode(state_prompt, return_tensors='pt')
            state_output = self.base_model(input_ids)
            valfunc_inputs = state_output  # Take the embedding as input
            inputs.append({'input_ids': valfunc_inputs})
        return inputs
    
    def __len__(self):
        return len(self.states)

    def get_raw_data(self, idx):
        return {
            'state': torch.tensor(self.states[idx], dtype=torch.float32),
            'action': torch.tensor(self.actions[idx], dtype=torch.long),
            'next_state': torch.tensor(self.next_states[idx], dtype=torch.float32),
            'done': torch.tensor(self.dones[idx], dtype=torch.bool),
            'predicted_q_value': self.predicted_q_values[idx], 
            'target_q_value': torch.tensor(self.target_q_values[idx], dtype=torch.float32)
        }
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(),
            'labels': self.target_q_values[idx]
        }

class QLearning:
    def __init__(self, foundation_model, val_func, tokenizer, env, num_samples):
        self.model = foundation_model
        self.val_func = val_func
        self.tokenizer = tokenizer
        self.env = env
        self.learning_scheme = ValueFunctionLearningScheme(self.model)
        self.datasets = []  # List to store multiple datasets
        self.current_dataset, self.states = self.update_dataset(num_samples)  # Initialize with the first dataset
        self.rewards = []  # Store rewards for each episode\

    def update_dataset(self, num_samples):
        # Create a new dataset with the specified number of samples
        new_dataset = QLearningDataset(self.model, self.val_func, self.tokenizer, self.env, num_samples)
        return new_dataset, new_dataset.states

    def learn(self):
        # Update the model using the training dataset
        training_data = self.current_dataset
        print("Training Data:", training_data)
        import pdb
        pdb.set_trace()
        ds = Dataset.from_list(training_data)
        self.learning_scheme.update_model(training_data)

    def plot_rewards(self, window_size=50):
        if len(self.rewards) < window_size:
            print("Not enough data to plot.")
            return

        # Calculate rolling average of the rewards
        rolling_avg = np.convolve(self.rewards, np.ones(window_size) / window_size, mode='valid')

        # Correct the x-axis values for the rolling average plot
        x_values = np.arange(window_size - 1, len(self.rewards))

        # Plotting the rewards and the rolling average
        plt.figure(figsize=(10, 5))
        plt.plot(x_values, rolling_avg, label='Rolling Average', color='red')  # Rolling average line
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Rewards Over Time During Training')
        plt.legend()
        plt.grid(True)
        plt.show()

class SimpleGridRewardFunction(AbstractRewardFunction):
    def __init__(self, goal_position=(2, 2)):
        self.goal_position = goal_position

    def compute_reward(self, next_state):
        # Check if the goal is reached
        if next_state.attributes['x'] == self.goal_position[0] and next_state.attributes['y'] == self.goal_position[1]:  # Assuming goal is at (2,2)
            return 100  # Reward for reaching the goal
        else:
            return -1  # Penalize each move to encourage shortest path

class SimpleEnvironment(AbstractEnvironment):
    def __init__(self, grid_size=4, context=None):
        self.grid_size = grid_size
        # Initialize with State instances using InputDict for positions
        self.current_state = State(attributes=InputDict({'x': 0, 'y': 0, 'orientation': 0}))
        self.reward_function = SimpleGridRewardFunction((self.grid_size-1, self.grid_size-1))  # Initialize the reward function
        self.done = False

    def reset(self):
        self.current_state.attributes['x'] = 0  # Reset position within the existing State instance
        self.current_state.attributes['y'] = 0
        return self.current_state
    
    def goal_criteria(self):
        # Goal is reached if the agent is at the bottom-right corner of the grid
        x, y = self.next_state.attributes['x'], self.next_state.attributes['y']
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
            print("Action:", action)

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
        self.next_state = State(attributes=InputDict({'x': new_x, 'y': new_y, 'orientation': current_orientation}))

        # Use the reward function to compute the reward
        reward = self.reward_function.compute_reward(self.next_state)
        self.done = self.goal_criteria()  # Check if goal is reached based on next_state

        return self.next_state, reward, self.done

    def get_current_state(self):
        return self.current_state  # Return the reference to the initial state
    
    def get_position(self):
        return self.current_state.attributes['x'], self.current_state.attributes['y']
    
    def update_state(self):
        self.current_state = self.next_state
    
    def close(self):
    # Add any necessary cleanup code here
        pass

class State:
    def __init__(self, attributes):
        self.attributes = attributes

    def to_tensor(self):
        return torch.tensor([self.attributes['x'], self.attributes['y'], self.attributes['orientation']], dtype=torch.float32)

class LactChainA(LactChain):
    def __init__(self,llm):
        super().__init__()
        self.strategy_component = self.create_strategy(llm)
        self.add_component(self.strategy_component)
        self.message = ""
        self.converted_strategy_component = self.convert_strategy(llm)
        self.add_component(self.converted_strategy_component)
        self.action_choices = []

    def add_feedback(self,message):
        context.update('feedback', message)

    def add_action(self, action):
        context.update('action_choices', action)

    def create_strategy(self,llm):
    
        class declare_strategy(Component):
            def __init__(self,llm):
                self.llm = llm
                self.move_prompt = dedent("""\
                    You are in gridworld. Make a move to help you reach the goal. 
                    Your response must be some kind of move, even if you have to guess. 
                    """)
            def execute(self,context=None):
                # Retrieve the last feedback from context, or use a default value if none is available
                # This didn't work very well - feedback mechanism needs to be rethought
        
                move_response = self.llm.invoke(self.move_prompt)
                print("Move Response:", move_response)
                self.message = move_response

        return declare_strategy(llm)

    def convert_strategy(self,llm):
        outer_message = self.message

        class ListOfMoves(BaseModel):
            moves: List[str]

        class convert_strategy_to_action(Component):
            def __init__(self,llm,parent):
                self.llm = llm
                self.parent = parent
                self.convert_prompt_template = dedent("""\
                    There are only 2 types of moves you can make:
            
                    1. move forward
                    2. turn left
            
                Come up with a combination of those two moves in order
                to successfully carry out the action: {strategy}
            
                Your final answer should be in the format of a python list 
                of moves, where each move is one of the 2 types listed above.
                E.g. ['move forward', 'turn left']
                """)
                # Fill in the placeholder in the prompt template
                self.prompt = self.convert_prompt_template.format(strategy=outer_message)

            def execute(self,context=None):
                print("Response from convert_strategy:", self.prompt)
                pydantic_parser = PydanticOutputParser(pydantic_object=ListOfMoves)
                format_instructions = pydantic_parser.get_format_instructions()

                #print("Prompt:", prompt)
                prompt_with_instructions = f"{self.prompt}\n\nFormat instructions: {format_instructions}"

                # LLM responds to prompt with instructions on how to move
                response = self.llm.invoke(prompt_with_instructions)
                #print("Response:", response)

                try:
                    # Parse the LLM response into the Pydantic model
                    parsed_response = pydantic_parser.parse(response)
                    print("Parsed Moves:", parsed_response.moves)
                    # Add the parsed moves to the action_choices list in the outer class
                    self.parent.action_choices.append(parsed_response.moves)
                except OutputParserException as e:
                    print("Failed to parse response:", e)

        return convert_strategy_to_action(llm,self)
    
    def propose_action(self):
        #propose_action should create a new action_choices list entry by calling create_strategy and convert_strategy
        create = self.create_strategy(llm)
        strategy = create.execute()
        print("Create Strategy", strategy)
        convert = self.convert_strategy(llm)
        action_list = convert.execute()
        print("Convert Strategy", action_list)
        return self.action_choices[-1] if self.action_choices else None #[-1] removed for testing

if __name__ == "__main__":
    # Example usage
    #llama_client = LlamaClientWrapper()
    """ llm = Ollama(model="llama3")
    prompt = "Translate the following English text to French: | Hello, how are you?"
    response = llm.invoke(prompt)
    print("Response from Llama Client:", response)

    #Test of State
    attributes = InputDict({'x': 0, 'y': 0})
    state = State(attributes=attributes, textblock="some text")
    print("State:", state.textblock)
    print("State Dict:", state.attributes)

    #Test of Environment class
    env = SimpleEnvironment(grid_size=4)
    initial_state = env.reset()
    print("Initial State:", initial_state)
    #lact_chain either makes an action or gets it from context['action_choices']
    lact_chain = LactChainA(llm=llm)
    lact_chain.execute()
    print("Action Choices 1:",lact_chain.action_choices)
    lact_chain.execute()
    print("Action Choices 2:",lact_chain.action_choices)
    print(lact_chain.action_choices[0])
    env.step(lact_chain.action_choices[0])
    env.step(lact_chain.action_choices[0])
    print("Final State after actions:", env.get_current_state().attributes)
    env.close()
 """
    #Test of LearningScheme on Gridworld
    llm = Ollama(model="llama3")
    env = SimpleEnvironment(grid_size=2)
    env.lact_chains = [LactChainA(llm=llm)]
    foundation_model = FoundationModel(model_name="mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = foundation_model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    val_func = ValueFunctionHead(foundation_model=foundation_model)
    qlearning = QLearning(foundation_model=foundation_model, val_func=val_func, tokenizer=tokenizer, env=env, num_samples=5)
    
    from torch.utils.data import DataLoader

    training_data = qlearning.current_dataset
    dataloader = DataLoader(training_data, batch_size=1, shuffle=False)
    print("Length of DataLoader:", len(dataloader))
    for batch in dataloader:
        print("Batch:", batch)
        exit(0)

    for _ in range(2):
        qlearning.learn()
        exit(0)
        qlearning.update_dataset(4)
    qlearning.plot_rewards()
    exit(0)
    
