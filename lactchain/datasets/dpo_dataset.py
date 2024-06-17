from torch.utils.data import Dataset, DataLoader
import torch
from collections import deque, namedtuple
from typing import Tuple, List, Any, Dict, Union, Callable, Optional
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset, concatenate_datasets
from torch import Tensor

class DPODataset(HFDataset): 
    '''
    NOTE: THIS SHOULD STORE THE TRANSITIONS, THEN WE MAKE ANOTHER CLASS THAT TURNS THIS INTO A
    HFDATASET VIA FROM_GENERATOR OR FROM_DICT
    This should be a dataset that stores the transition + the advantage that you calculate at the 
    End of the episode
    '''
    def __init__(self):
        super().__init__() 
        self.dataset=HFDataset.from_dict(
            {'state':[], 'action':[], 'reward':[], 'next_state':[], 'advantage':[]}
            )
        self.transition=namedtuple('Transition', 
                            ['state', 'action', 'reward', 'next_state', 'advantage']
                            )

    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, idx:int): 
        return self.dataset[idx]

    def add_batch_transitions(self, states:Tensor, actions:list[int], 
                              rewards:Tensor, next_states:Tensor, 
                              advantages:Tensor) -> None:
        _transition=self.transition(states, actions, rewards, next_states, advantages)
        _new_dataset=HFDataset.from_dict(_transition._asdict())
        self.dataset=concatenate_datasets([self.dataset, _new_dataset])
        
    def __str__(self): 
        return f'{self.dataset}'


class DPOCollator(object): 
    def __init__(self, tokenizer:Callable, call_kwargs:Optional[Dict[str, Any]]=None): 
        from transformers import AutoTokenizer
        self.tokenizer=tokenizer
        if tokenizer.pad_token is None: 
            self.tokenizer.pad_token=self.tokenizer.eos_token
        self.call_kwargs=call_kwargs if call_kwargs is not None else \
            {'padding':'longest', 
             'return_tensors':'pt'}

    def __call__(self, batch:list[str]) -> list[str]:

        batch = [self.tokenizer(prompt, **self.call_kwargs) 
                 for prompt in batch]
        return batch

    
if __name__=="__main__": 
    import torch, torch.nn as nn, torch.nn.functional as F
    import sys, os

    dataset=DPODataset()
    r, gamma = 0.1, 0.99

    V = nn.Linear(1, 1)

    state={'x':3, 'y':0, 'orientiation':100}
    action=['move left', 'move right']
    reward=-6
    next_state={'x':1, 'y':10, 'orientation':200}
    advantage=1000
    breakpoint()
    advantage=reward + gamma*V(torch.Tensor(next_state['x'],)) + V(torch.Tensor(state['x'],))
    
    dataset.add_transition(state, action, reward, next_state, advantage)

    breakpoint()




# class QLearningDataset(Dataset):
#     def __init__(self, model, env, num_samples, gamma=0.99):
#         self.model = model
#         self.env = env
#         self.gamma = gamma
#         self.states = []
#         self.actions = []
#         self.next_states = []
#         self.dones = []
#         self.target_q_values = []
#         self.max_steps = 10
#         #getting target and predicted q-values, respectively
#         self.sample_experience(num_samples)
#         self.predicted_q_values = self.get_predicted_q_values()  # Compute predicted Q-values for all states

#     def sample_experience(self, num_samples):
#         for _ in range(num_samples):
#             state = self.env.reset()  # Start a new episode
#             done = False
#             steps = 0
#             while not done:
#                 action = self.env.lact_chains[0].propose_action()  # Get action from the environment's LactChain
#                 next_state, reward, done = self.env.step(action)  # Execute the action

#                 self.states.append(state)
#                 self.actions.append(action)
#                 self.next_states.append(next_state)
#                 self.dones.append(done)

#                 # Compute target Q-value for this transition
#                 target_q_value = self.compute_target_q_value(next_state, reward, done)
#                 self.target_q_values.append(target_q_value)

#                 state = next_state  # Move to the next state
#                 steps += 1
#                 if steps > self.max_steps:
#                     done = True

#     def compute_target_q_value(self, next_state, reward, done):
#         if done:
#             return reward
#         else:
#             self.model.eval()
#             with torch.no_grad():
#                 # Predict Q-values for the next state
#                 next_state_tensor = next_state.to_tensor().unsqueeze(0)
#                 next_state_q_values = self.model(next_state_tensor)
#                 max_next_q_value = torch.max(next_state_q_values).item()
#             self.model.train()
#             return reward + self.gamma * max_next_q_value

#     def get_predicted_q_values(self):
#         self.model.eval()
#         with torch.no_grad():
#             states_tensor = torch.stack([torch.tensor(state, dtype=torch.float32) for state in self.states])
#             predicted_q_values = self.model(states_tensor)
#         self.model.train()
#         return predicted_q_values
    
#     def __len__(self):
#         return len(self.states)

#     def __getitem__(self, idx):
#         return {
#             'state': torch.tensor(self.states[idx], dtype=torch.float32),
#             'action': torch.tensor(self.actions[idx], dtype=torch.long),
#             'next_state': torch.tensor(self.next_states[idx], dtype=torch.float32),
#             'done': torch.tensor(self.dones[idx], dtype=torch.bool),
#             'predicted_q_value': self.predicted_q_values[idx], 
#             'target_q_value': torch.tensor(self.target_q_values[idx], dtype=torch.float32)
#         }