from abc import ABC, abstractmethod

class LearningScheme(ABC):
    """
    Abstract base class for learning schemes used to update networks based on rewards.
    This class defines the interface for training and updating models in a reinforcement learning system.
    """
    
    @abstractmethod
    def update_model(self, state, action, reward, next_state, done):
        """
        Update the model based on the state, action, reward, next state, and done signal.
        
        Parameters:
            state: The current state from the environment.
            action: The action taken by the agent.
            reward: The reward received from the environment after taking the action.
            next_state: The state of the environment after the action is taken.
            done: A boolean flag indicating whether the episode has ended.
        
        This method should implement the logic for updating the model based on the learning algorithm.
        """
        pass

    @abstractmethod
    def save_model(self, filepath):
        """
        Save the model to a specified file path.
        
        Parameters:
            filepath: The path where the model should be saved.
        """
        pass

    @abstractmethod
    def load_model(self, filepath):
        """
        Load the model from a specified file path.
        
        Parameters:
            filepath: The path from where the model should be loaded.
        """
        pass