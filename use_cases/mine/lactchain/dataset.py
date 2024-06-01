from torch.utils.data import Dataset, DataLoader
import torch
from collections import deque, namedtuple
from typing import Tuple, List, Any, Dict, Union
from dataclasses import dataclass
from datasets import 

import sys, os
sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')


class Memory(object): 
    def __init__(self, max_len:int): 
        self.storage=deque(maxlen=max_len)
        self.Transition=namedtuple('Transition', 
                                   ['state', 'action', 'reward', 'next_state']
                                   )

    def add_transition(self, 
                       state:Dict[str, Any], 
                       action:List[str], 
                       reward:int,
                       next_state:Dict[str, Any], 
                       ):
        transition=self.Transition(state, action, reward, next_state)
        self.storage.append(transition)
    
    def __len__(self): 
        return len(self.storage)
    
    def __getitem__(self, idx:int) -> Any:
        return self.storage[idx]
    
class Dataset(): 
    '''This should be a dataset that stores the transition + the advantage that you calculate at the 
    End of the episode
    '''
    ...
    
if __name__=="__main__": 

    memory=Memory(100)

    state={'x':0, 'y':0, 'orientiation':100}
    action=['move left', 'move right']
    reward=-6
    next_state={'x':1, 'y':10, 'orientation':200}
    advantage=1000
    
    memory.add_transition(state, action, reward, next_state, advantage)

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