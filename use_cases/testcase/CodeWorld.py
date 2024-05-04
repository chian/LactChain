from abc import ABC
import subprocess
import sys, os

print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)

sys.path.append('/Users/chia/Documents/ANL/Software/LactChain/')
from classes.state import State
from classes.reward import AbstractRewardFunction
from classes.environment import AbstractEnvironment

class CodeWorldRewardFunction(AbstractRewardFunction):
    def compute_reward(self, current_state, action, next_state):
        # Example reward logic based on output correctness
        if "success" in next_state:
            return 100  # Reward for achieving the goal
        return -10  # Penalty for not achieving the goal

class CodeWorld(AbstractEnvironment):
    def __init__(self):
        super().__init__()
        self.action_space = []  # Define specific actions if necessary
        self.observation_space = None  # Define observation space details
        self.current_state = None
        self.reward_function = CodeWorldRewardFunction()
        self.low_privilege_user = 'sandboxuser'  # Username of the low-privileged user

    def step(self, action):
        # Command to execute Python code as a different user
        #command = f"sudo -u {self.low_privilege_user} {sys.executable} -c \"{self.current_state.textblock}\""
        command = f"{sys.executable} -c \"{self.current_state.textblock}\""
        try:
            output = subprocess.run(
                command,
                shell=True,  # Use shell to interpret the sudo command
                capture_output=True,
                text=True,
                check=True
            )
            new_state = output.stdout
            reward = self.reward_function.compute_reward(self.current_state.textblock, action, new_state)
            done = self.goal_criteria(new_state)
            return (new_state, reward, done)
        except subprocess.CalledProcessError as e:
            print(f"Error executing action: {e}")
            return (self.current_state, 0, True)  # No reward, episode ends

    def reset(self):
        # Reset the environment to an initial state
        self.current_state = State(textblock="print('Hello, CodeWorld')")
        return self.current_state

    def close(self):
        # Clean up resources if necessary
        pass

    def goal_criteria(self, state):
        # Define goal criteria, for example, checking output correctness
        return "success" in state
    
if __name__ == '__main__':
    # Initialize the CodeWorld environment
    env = CodeWorld()
    env.current_state = State(textblock="print('Hello, CodeWorld')")

    # Execute the step method
    action = None  # Assuming no specific action is needed for this test
    new_state, reward, done = env.step(action)

    # Output the results to verify behavior
    print("Output:", new_state)
    print("Reward:", reward)
    print("Done:", done)