import sys, os
from gymnasium import spaces
import gymnasium as gym
import numpy as np
sys.path.append(os.getcwd()+'/../../../')
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
    
    
    
    


class GridEnvironment2(gym.Env): 
    def __init__(self, 
                 grid_size: int = 4, 
                 goal_position: Tuple[int, int] = (1, 1),
                 context: Any = None, 
                 render_mode: str = None
                 ): 
        '''Our Gridworld environment: 
        Action Space (2): [move forward, turn left]
        Orientation ()
        '''
        super().__init__()
        self._grid_size = grid_size
        self.num_orientations = 4 # assume number of orientations is 4
        
        self.action_space = spaces.Discrete(2) # action space: move forward or turn left 
        self.observation_space = spaces.Dict(
            {
                'x': spaces.Discrete(self.grid_size),
                'y': spaces.Discrete(self.grid_size),
                'orientation': spaces.Discrete(self.num_orientations)
            }
        )

        self.orientation_set_probability = np.ones(self.num_orientations) / self.num_orientations
        self.goal_position = goal_position
        self.state = None
        self.reset()
        
    @property
    def grid_size(self) -> int: 
        return self._grid_size 
        
    @grid_size.setter
    def grid_size(self, grid_size: int) -> None: 
        self._grid_size = grid_size
        
    @property
    def coord_set_probability(self) -> np.ndarray: 
        return np.ones(self._grid_size) / self._grid_size

    def reset(self) -> Dict[str, int]: 
        self.state = {'x': np.array(0), 'y': np.array(0), 'orientation': np.array(0)} # resetting state
        _state={k:np.array(v) for k,v in self._get_obs().items()}
        return _state

    def _get_obs(self) -> Dict[str, int]: 
        return {'x': self.state['x'], 
                'y': self.state['y'], 
                'orientation': self.state['orientation']}
    
    def step(self, action_sequence: List[int]) -> Tuple[Dict[str, int], int, bool, Dict[str, Any]]:
        total_reward = 0
        done = False

        for action in action_sequence:
            if done:
                break
            self.state, reward, done = self._process_action(action)
            total_reward += reward

        _states={k:np.array(v) for k,v in self._get_obs().items()}
        return _states, np.array(total_reward), np.array(done), {}

    def _process_action(self, action: int) -> Tuple[Dict[str, int], int, bool]:

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
            orientation = (orientation - 1) % 4 # a % b = a - floor(a / b) * b

        # Enforce boundary conditions
        x = max(0, min(x, self.grid_size - 1))
        y = max(0, min(y, self.grid_size - 1))

        # Update state and get reward and goal state
        self.state = {'x': np.array(x), 'y': np.array(y), 'orientation': np.array(orientation)}
        reward = self._compute_reward()
        done = self._goal_criteria()

        return self._get_obs(), reward, done
    
    def _compute_reward(self) -> int:
        if (self.state['x'], self.state['y']) == self.goal_position:
            return np.array(100)
        else:
            return np.array(-1) # penalty if no finish

    def _goal_criteria(self) -> bool:
        return (self.state['x'], self.state['y']) == self.goal_position   
    

    
class FuckingSimpleEnv(gym.Env): 
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
        
    def reset(self) -> Dict[str, np.ndarray]: 
        self.state = {'x': 0, 'y': 0, 'orientation': 0}
        return self.state, {'info':f'information'}
        
    def step(self, action:Any):
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

        reward = 0  # set to one value for now
        done = False  # set your own condition for done

        self.state = {'x': x, 'y': y, 'orientation': orientation}

        return self.state, reward, done, {'info':f'information'}
    
    
    
    
####################### testing ground for 
def ProcessExecBasic(): 
    ...
        
def main(): 
    # import torch, import torch.distributed as dist, import torch.multiprocessing as mp
    
    world_size=torch.cuda.device_count()
    num_processes=world_size
    processes=[]
    backend='nccl'
    mp.set_start_method('spawn')

    for rank in range(num_processes): 
        process=mp.Process(target=dist_train, args=(world_size, rank, backend))
        process.start()
        processes.append(process)

    for process in processes: 
        process.join()
        




if __name__=="__main__": 
    
    from gymnasium.vector import AsyncVectorEnv
    
    env=FuckingSimpleEnv()
    num_envs=4
    
    def make_envs(): 
        return FuckingSimpleEnv()
    
    dist_env = AsyncVectorEnv([make_envs for _ in range(num_envs)])
    
    out, info = dist_env.reset()
    
    actions=dist_env.action_space.sample()
    breakpoint()
    next_obs, reward, done, info = dist_env.step(actions) 
    breakpoint()
    
    
    sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')
    from use_cases.mine.lactchain.dataset import QLearningDataset

    cum_rewards=[]
    policy_losses=[]
    critic_losses=[]

    env=GridEnvironment()
    obs = env.reset()
    action_seq=['turn left', 'turn left', 'move forward', 'move forward', 'move forward', 'move forward']
    obs, tot_reward, rewards, done = env.step(action_seq)

    values=[1, 1, 1, 1, 1]
    
    breakpoint()