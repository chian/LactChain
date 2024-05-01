class AgentInterface:
    """
    A custom interface or wrapper for agents interacting with the environment.
    This can enforce certain checks or requirements on the agent's policy and value functions.
    """
    
    def __init__(self, agent, environment):
        self.agent = agent
        self.environment = environment
    
    def check_compliance(self):
        """
        Implement checks to ensure the agent's policy and value functions are compliant.
        """
        # Example check
        if not hasattr(self.agent, 'policy'):
            raise AttributeError("Agent must have a policy attribute.")
        # Add more checks as needed
    
    def run_episode(self):
        """
        Run an episode with the agent, using the environment.
        """
        self.check_compliance()
        state = self.environment.reset()
        done = False
        while not done:
            action = self.agent.policy(state)
            state, reward, done, info = self.environment.step(action)
            # Implement the logic for interacting with the environment