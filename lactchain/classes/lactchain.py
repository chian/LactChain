class LactChain:
    """
    LactChain class.
    """
    def __init__(self, state: State):
        self.state = state

    def transition(self, action, *args, **kwargs):
        """
        I asked GPT4 to help me code this as a base class that will later
        be specified by the user. I think the answer is hard code them
        yourself later.

        Transition the state based on an action or set of actions that
        define the language action chain. Implement this method in subclasses
        to specify the behavior of your LactChain. This method should take an
        action, apply it to the current state, and return the new state.
        
        Parameters:
        - action: The action to be applied. This could be a function or any
                  other callable that takes the current state (and optionally
                  other arguments) and returns a new state.
        - *args, **kwargs: Additional arguments and keyword arguments that
                           might be needed for the action.
        
        Returns:
        - A new state resulting from applying the action to the current state.
        
        Example:
        ```
        def my_action(state, *args, **kwargs):
            # Modify the state or compute a new state
            new_state = ...
            return new_state
        
        # In a subclass of LactChain
        def transition(self, action, *args, **kwargs):
            return action(self.state, *args, **kwargs)
        ```
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    # Specify individual actions to be used in the language action chain.
    # Do this in subclasses built from this class.
    # These can be CoT, agent calls, prompt mutations, etc.
    # If prompts are stored in the state for later use (i.e., your action is
    # just to mutate a prompt) then you can update the state dictionary 
    # (which does not modify the embedding) and return a state with the same
    # embedding (but with dictionary modified).

    # Does this class need any other methods?

