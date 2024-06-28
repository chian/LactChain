import torch
from gymnasium import spaces
import gymnasium as gym
import numpy as np
from lactchain.classes.base_environment import AbstractEnvironment
from lactchain.classes.base_reward import AbstractRewardFunction
from typing import Tuple, Any, Dict, List
from torch.distributions import Categorical

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

        self.orientation_set_probability=np.ones(self.num_orientations) / self.num_orientations
        self.goal_position=goal_position
        self.state=None
        self.reset()
        
    @property
    def info(self) -> str: 
        return f'Grid is size {self.grid_size}, goal position is at {self.goal_position}'

    @property
    def grid_size(self) -> int: 
        return self._grid_size 
        
    @grid_size.setter
    def grid_size(self, grid_size:int) -> None: 
        self._grid_size=grid_size
        
    @property
    def coord_set_probability(self) -> np.ndarray: 
        return np.ones(self._grid_size) / self._grid_size
    
    @property
    def coordinate_space_distro(self) -> Categorical: 
        coord_space_prob=torch.from_numpy(self.coord_set_probability)
        return torch.distributions.Categorical(coord_space_prob)
    
    @property 
    def orientation_set_distro(self) -> Categorical: 
        orientation_space_prob=torch.from_numpy(self.orientation_set_probability)
        return torch.distributions.Categorical(orientation_space_prob)

    def reset(self) -> Tuple[Dict[str, Any], str]: 
        self.state={'x':0, 'y':0, 'orientation':0} # ressetting state
        return self._get_obs(), f'Grid is size {self.grid_size}, goal position is at {self.goal_position}'

    def _get_obs(self) -> Dict[str, Any]: 
        return {'x':self.state['x'], 
                'y':self.state['y'], 
                'orientation':self.state['orientation']}
    
    def step(self, action_sequence:List[str]) -> Tuple[Dict[str, int], 
                                                       int, 
                                                       bool, 
                                                       str]:
        total_reward = 0
        done = False

        for action_command in action_sequence:
            if done:
                break
            self.state, reward, done = self._process_action(action_command)
            total_reward += reward

        return self._get_obs(), total_reward, done, f'Grid is size {self.grid_size}, goal position is at {self.goal_position}'
    
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
    
    
class VectorizedGridWorld(gym.Env): 
    def __init__(self, 
                 grid_size: int = 4, 
                 goal_position: Tuple[int, int] = (1, 1),
                 num_orientations: int = 4,
                 context: Any = None, 
                 render_mode: str = None):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_orientations = num_orientations
        self.action_space = spaces.Discrete(2)  # action space: move forward or turn left 
        self.observation_space = spaces.Dict(
            {
                'x': spaces.Discrete(self.grid_size),
                'y': spaces.Discrete(self.grid_size),
                'orientation': spaces.Discrete(self.num_orientations)
            }
        )
        self.goal_position = goal_position
        self.context = context
        self.render_mode = render_mode
        self.state = None
        
        self.coordinate_set_probability=np.ones(self.grid_size) / self.grid_size
        self.orientation_set_probability=np.ones(self.num_orientations) / self.num_orientations
        
    @property
    def coordinate_space_distro(self) -> Categorical: 
        coord_space_prob=torch.from_numpy(self.coordinate_set_probability)
        return torch.distributions.Categorical(coord_space_prob)
        
    @property 
    def orientation_set_distro(self) -> Categorical: 
        orientation_space_prob=torch.from_numpy(self.orientation_set_probability)
        return torch.distributions.Categorical(orientation_space_prob)
        
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]: 
        self.state = {'x': 0, 'y': 0, 'orientation': 0}
        return self.state, {'info': 'information'}
        
    def step(self, actions: Any) -> Tuple[Dict[str, int], float, bool, bool, Dict[str, str]]:
        total_reward = 0
        for action in actions:
            assert action in [0, 1], 'Invalid Action: Must Choose from [0, 1]'
            x, y, orientation = self.state['x'], self.state['y'], self.state['orientation']
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
                orientation = (orientation - 1) % 4  # a % b = a - floor(a / b) * b

            # Enforce boundary conditions
            x = max(0, min(x, self.grid_size - 1))
            y = max(0, min(y, self.grid_size - 1))

            self.state = {'x': x, 'y': y, 'orientation': orientation}
            total_reward += self._compute_reward()

        # done = (self.state['x'], self.state['y']) == self.goal_position
        done = (self.state['x'], self.state['y']) == (0, 0)
        truncated = False  # set your own condition for truncated if needed
        return self.state, total_reward, done, truncated, {'info': 'information'}
    
    def _compute_reward(self) -> int:
        if (self.state['x'], self.state['y']) == self.goal_position:
            return 100
        else:
            return -1  # penalty if not finished

def make_env():
    return VectorizedGridWorld()


if __name__=="__main__": 
    
    out=VectorizedGridWorld()
    distro=out.orientation_set_distro
    other_distro=out.coordinate_space_distro
    num_envs = 4  # Number of environments to run in parallel
    async_vector_env = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])
    
    # for env in async_vector_env.envs:
    #     print(env.orientation_set_distro)
    
    # obs, info = async_vector_env.reset()

    # actions=[np.array([0, 1]), np.array([1, 0, 1, 0, 1, 0]), np.array([0, 1]), np.array([1, 0])]
    # next_obs, reward, done, truncated, info = async_vector_env.step(actions)
    
    breakpoint()
    
    # obs = env.reset()
    # action_seq=['turn left', 'turn left', 'move forward', 'move forward', 'move forward', 'move forward']
    # obs, tot_reward, rewards, done = env.step(action_seq)

    # values=[1, 1, 1, 1, 1]
    
    breakpoint()