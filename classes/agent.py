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

def learn(self, state, action, reward, next_state, done):
        """
        Delegate the learning process to the learning scheme.
        """
        self.learning_scheme.update_model(state, action, reward, next_state, done)