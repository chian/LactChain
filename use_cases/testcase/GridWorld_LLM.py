import os
print("Current Working Directory:", os.getcwd())
import sys
print("Python Path:", sys.path)
sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')

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
from transformers import TrainingArguments, Trainer, AutoModel, AutoTokenizer

class LlamaClientWrapper:
    def __init__(self):
        self.client = Client(host='http://localhost:11434')

    def invoke(self, prompt):
        response = self.client.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']

class ValueFunction(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        super(ValueFunction, self).__init__()
        #ADD LORA CONFIG HERE
        self.bb_config = None
        self.use_qlora = False
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            # target modules varies from model to model
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )
        self.printer = True
        self.max_seq_length = 128
        self.model_name = model_name
        self.tokenizer_name = "LlamaTokenizer"
        
        # Load the model        
        if self.use_qlora:
            from peft import prepare_model_for_kbit_training
            from transformers import BitsAndBytesConfig
            bb_config = BitsAndBytesConfig(
                load_in_8_bit=True,
            )
        if self.printer:
            print('Loading model:', self.model_name)
        self.base_model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            #attn_implementation="flash_attention_2",
            #quantization_config=bb_config
        )
        if self.use_qlora:
            if self.printer:
                print('Preparing model for PEFT training')
            self.model = prepare_model_for_kbit_training(self.base_model, self.lora_config)
            if self.printer:
                print('Model prepared for PEFT training')
        # Use PEFT, if requested
        if self.lora_config is not None:
            if self.printer:
                print('setting up peft!')
            #peft_config = get_peft_config(self.peft_config)
            self.model = get_peft_model(self.base_model, self.lora_config)
            if self.printer:
                self.model.print_trainable_parameters()
        if self.printer:
            print('Model:', self.model)
        self.q_value_head = nn.Linear(self.model.config.hidden_size, 1)
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(self.tokenizer_name,)
        # Set the model max length for proper truncation;
        # for mistral, 32k is too long and causes OOM failures
        self.tokenizer.model_max_length = min(self.model.config.max_position_embeddings,
                                        self.max_seq_length)
        # Set the pad token if it is not set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = '[PAD]'

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        q_values = self.q_value_head(last_hidden_states[:, 0, :])  # Using the first token's representation
        predicted_q_value = q_values.mean()  # Take the mean of the first logit
        return predicted_q_value

class ValueFunctionLearningScheme(LearningScheme):
    def __init__(self, model):
        self.model = model
        self.trainer = None
        self.training_args = TrainingArguments(
            output_dir="./model_output",
            num_train_epochs=1,  # Set to 1 for episodic training
            per_device_train_batch_size=16,
            learning_rate=1e-5,  # Adjusted for PEFT
            logging_dir='./logs',
            logging_steps=10,
        ) 

    def update_model(self, train_dataset):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            compute_loss=self.compute_loss
        )
        self.trainer.train()

    def compute_loss(self,batch):
        # Use the precomputed predicted Q-values from the dataset
        predicted_q_values = batch['predicted_q_value']

        # Extract the target Q-values from the batch
        target_q_values = batch['target_q_value']

        # Compute the mean squared error loss
        loss = torch.nn.functional.mse_loss(predicted_q_values, target_q_values)
        return loss
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))

class QLearningDataset(torch.utils.data.Dataset):
    def __init__(self, model, env, num_samples, gamma=0.99):
        self.model = model
        self.env = env
        self.gamma = gamma
        self.states = []
        self.actions = []
        self.next_states = []
        self.dones = []
        self.target_q_values = []
        self.max_steps = 10
        #getting target and predicted q-values, respectively
        self.sample_experience(num_samples)
        self.predicted_q_values = self.get_predicted_q_values()  # Compute predicted Q-values for all states

    def sample_experience(self, num_samples):
        for _ in range(num_samples):
            state = self.env.reset()  # Start a new episode
            done = False
            steps = 0
            while not done:
                action = self.env.lact_chains[0].propose_action()  # Get action from the environment's LactChain
                next_state, reward, done = self.env.step(action)  # Execute the action

                self.states.append(state)
                self.actions.append(action)
                self.next_states.append(next_state)
                self.dones.append(done)

                # Compute target Q-value for this transition
                target_q_value = self.compute_target_q_value(next_state, reward, done)
                self.target_q_values.append(target_q_value)

                state = next_state  # Move to the next state
                steps += 1
                if steps > self.max_steps:
                    done = True

    def compute_target_q_value(self, next_state, reward, done):
        if done:
            return reward
        else:
            self.model.eval()
            with torch.no_grad():
                # Predict Q-values for the next state
                next_state_tensor = next_state.to_tensor().unsqueeze(0)
                next_state_q_values = self.model(next_state_tensor)
                max_next_q_value = torch.max(next_state_q_values).item()
            self.model.train()
            return reward + self.gamma * max_next_q_value

    def get_predicted_q_values(self):
        self.model.eval()
        with torch.no_grad():
            states_tensor = torch.stack([torch.tensor(state, dtype=torch.float32) for state in self.states])
            predicted_q_values = self.model(states_tensor)
        self.model.train()
        return predicted_q_values
    
    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'state': torch.tensor(self.states[idx], dtype=torch.float32),
            'action': torch.tensor(self.actions[idx], dtype=torch.long),
            'next_state': torch.tensor(self.next_states[idx], dtype=torch.float32),
            'done': torch.tensor(self.dones[idx], dtype=torch.bool),
            'predicted_q_value': self.predicted_q_values[idx], 
            'target_q_value': torch.tensor(self.target_q_values[idx], dtype=torch.float32)
        }

class QLearning:
    def __init__(self, model, env, num_samples):
        self.model = model
        self.env = env
        self.learning_scheme = ValueFunctionLearningScheme(self.model)
        self.datasets = []  # List to store multiple datasets
        self.update_dataset(num_samples)  # Initialize with the first dataset
        self.rewards = []  # Store rewards for each episode

    def update_dataset(self, num_samples):
        # Create a new dataset with the specified number of samples
        new_dataset = QLearningDataset(self.model, self.env, num_samples)
        self.datasets = [new_dataset]

    def learn(self):
        # Update the model using the training dataset
        self.learning_scheme.update_model(self.dataset)

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
                    <s>[INST] You are in gridworld. Make a move to help you reach the goal. 
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
                [INST]
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
        self.create_strategy(llm)
        self.convert_strategy(llm)
        return self.action_choices #[-1] removed for testing

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
    env = SimpleEnvironment(grid_size=3)
    env.lact_chains = [LactChainA(llm=llm)]
    val_func = ValueFunction(model_name="mistralai/Mistral-7B-Instruct-v0.2")
    print(val_func.model)
    qlearning = QLearning(model=val_func.model, env=env, num_samples=4)
    for _ in range(10):
        qlearning.learn()
        qlearning.update_dataset(4)
    qlearning.plot_rewards()
    exit(0)
    
