#State Class Template

# This class allows you to define whatever information you will track in state, but not make
# part of the state object itself. This is useful for tracking information that is not
# directly related to the state of the system, but is needed for the state transition function
# and operates in a non-hard coded way.
class InputDict:
    """
    A simple dictionary-like class that allows attributes to be set and accessed using dictionary keys.
    """
    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)

    def __getitem__(self, key):
        return self.__dict__.get(key)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __repr__(self):
        return str(self.__dict__)

class State(InputDict):
    """
    This class defines the state operations, assuming the state will be an LLM embedded block of text.
    It extends InputDict to utilize a flexible attribute setting and accessing mechanism.
    """
    def __init__(self, iterable=(), **kwargs):
        self.embedding_model = kwargs.pop('embedding_model', None)
        self.textblock = kwargs.pop('textblock', None)
        super().__init__(iterable, **kwargs)
        self.embedding = None

    def __call__(self):
        """
        Invoke the embedding model on the textblock to compute and return the embedding.
        """
        if self.embedding_model and self.textblock:
            self.embedding = self.embedding_model.invoke(self.textblock)
        return self.embedding

    def validate_textblock(self, format_spec):
        """
        Validate the textblock against a specified format.
        """
        # Example validation logic (to be implemented based on specific requirements)
        if not isinstance(self.textblock, str):
            raise ValueError("Textblock must be a string.")
        # Additional format-specific validation can be added here

    def extract_information(self, extractor_type):
        """
        Extract specific information from the textblock using a defined extractor type.
        """
        # Example extractor logic (to be implemented based on specific requirements)
        if extractor_type == 'code':
            # Extract code snippets from textblock
            pass
        elif extractor_type == 'questions':
            # Extract questions from textblock
            pass
        # Additional extractors can be defined here

    # Add any extractors
    # example, if you want to extract code from a textblock, write a function to do that
    # here. Same if you wanted to extract out questions from a textblock. These 
    # extractors likely need to be hardcoded as functions. 
    # 

    
    # If the textblock
    # has a format specification that is passed to the validator, that should be
    # used to extract the relevant information here by specifying which kind of 
    # extractor to use for which aspect of the textblock.
        

