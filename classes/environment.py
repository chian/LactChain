from abc import ABC, abstractmethod

class AbstractEnvironment(ABC):
    """
    Abstract class for creating custom environments for reinforcement learning or other simulations.
    This class defines the basic structure and required methods that must be implemented by any concrete subclass.
    """
    def __init__(self):
        super().__init__()
        self.action_space = None  # Define in subclass
        self.observation_space = None  # Define in subclass

    @abstractmethod
    def step(self, action):
        """
        Apply the action to the environment and return the resulting state, reward, and done status.
        
        Parameters:
            action: An action to be applied in the environment.
        
        Returns:
            tuple: Contains the new state, the reward for the action, and a boolean indicating if the episode is complete.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the environment to an initial state and return the initial observation.
        
        Returns:
            initial_state: The state of the environment at the start of a new episode.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Perform any necessary cleanup at the end of an environment's life cycle.
        """
        pass