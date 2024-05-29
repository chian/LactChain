import sys, os
from gymnasium import spaces
import gymnasium as gym

sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain')
from classes.environment import AbstractEnvironment
from classes.reward import AbstractRewardFunction
from typing import Tuple, Any, Dict, List

class GridEnvironment(gym.Env): 
    def __init__(self, 
                 grid_size:int=4, 
                 goal_position:Tuple[int, int]=(1, 1),
                 context:Any=None, 
                 render_mode:str=None
                 ): 
        '''Our Gridworld environment: 
        Action Space (2): [move forward, turn left]
        Orientation ()
        '''
        self.grid_size=grid_size
        self.action_space=spaces.Discrete(2) # action space: move forward or turn left 
        self.num_orientations=4 # assume number of orientations is 4
        self.observation_space=spaces.MultiDiscrete([self.grid_size, 
                                                    self.grid_size, 
                                                    self.num_orientations]) # 
        self.goal_position=goal_position
        self.state=None
        self.reset()

    def reset(self) -> Tuple[int]: 
        self.state={'x':0, 'y':0, 'orientation':0} # ressetting state
        return self._get_obs()

    def _get_obs(self) -> Tuple[Dict[str, int], Dict[str, Any]]: 
        return {'x':self.state['x'], 
                'y':self.state['y'], 
                'orientation':self.state['orientation']}
    
    def step(self, action_sequence:List[str]) -> Tuple[Tuple[int], int, bool]:
        total_reward = 0
        done = False

        for action in action_sequence:
            if done:
                break
            self.state, reward, done = self._process_action(action)
            total_reward += reward

        return self._get_obs(), total_reward, done
    
    def _process_action(self, action_command:str) -> Tuple[Tuple[int], int, bool]:
        assert action_command in ['move forward', 'turn left'], \
            f'Invalid Action: Must Choose from [move forward, turn left]'
        
        x, y, orientation = self.state['x'], self.state['y'], self.state['orientation']

        actions={
            'move forward':0, 
            'turn left':1
            }
        action=actions.get(action_command)
        if action == 0:  # move forward
            if orientation == 0:  # facing up
                y -= 1
            elif orientation == 1:  # facing right
                x += 1
            elif orientation == 2:  # facing down
                y += 1
            elif orientation == 3:  # facing left
                x -= 1
        elif action == 1:  # turn left
            orientation = (orientation - 1) % 4 # a % b = a - floor(a / b) * b

        # Enforce boundary conditions
        x = max(0, min(x, self.grid_size - 1))
        y = max(0, min(y, self.grid_size - 1))

        # Update state and get reward and goal state
        self.state = {'x': x, 'y': y, 'orientation': orientation}
        reward = self._compute_reward()
        done = self._goal_criteria()

        return self._get_obs(), reward, done
    
    def _compute_reward(self):
        if self.state['x'] == self.goal_position and self.state['y'] == self.goal_position:
            return 100
        else:
            return -1 # penalty if no finishe

    def _goal_criteria(self):
        return self.state['x'] == self.goal_position and self.state['y'] == self.goal_position
    

if __name__=="__main__": 

    env=GridEnvironment()
    obs = env.reset()
    action_seq=['turn left', 'turn left', 'move forward', 'move forward', 'move forward', 'move forward']
    obs = env.step(action_seq)
    breakpoint()