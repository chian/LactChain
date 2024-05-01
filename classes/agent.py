from abc import ABC, abstractmethod

class AbstractRLAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.
    This class defines the basic structure and required methods that must be implemented by any concrete subclass.
    """

    @abstractmethod
    def select_action(self, state):
        """
        Select an action based on the given state.
        
        Parameters:
            state: The current state from the environment.
        
        Returns:
            action: The action chosen by the agent.
        """
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        """
        Update the agent's knowledge or model based on the experience.
        
        Parameters:
            state: The current state from the environment.
            action: The action taken by the agent.
            reward: The reward received after taking the action.
            next_state: The state of the environment after the action is taken.
            done: A boolean flag indicating whether the episode has ended.
        """
        pass