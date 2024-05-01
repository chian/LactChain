from abc import ABC, abstractmethod

class AbstractRewardFunction(ABC):
    """
    Abstract class to calculate the reward given the current state, action, and next state.
    This class cannot be instantiated on its own and must be subclassed with specific reward calculations.
    """
    def __init__(self, config=None):
        """
        Initialize with optional configuration settings.
        """
        self.config = config or {}

    @abstractmethod
    def compute_reward(self, current_state, action, next_state):
        """
        Abstract method to compute and return the reward based on the current state, action taken, and the next state.
        
        Must be implemented by subclasses.
        
        Parameters:
            current_state: The state before the action was taken.
            action: The action taken by the agent.
            next_state: The state after the action was taken.
        
        Returns:
            A numeric reward.
        """
        pass
