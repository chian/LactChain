#State Class Template

# This class allows you to define whatever information you will track in state, but not make
# part of the state object itself. This is useful for tracking information that is not
# directly related to the state of the system, but is needed for the state transition function
# and operates in a non-hard coded way.
class InputDict(object):
    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)

# This class defines the state operations, assuming the state will be an LLM embedded block of 
# text.
class State(InputDict):

    def __init__(self, iterable=(), **kwargs):
        # set model and textblock variables before the other dictionary items if they are in
        # **kwargs.
        self.embedding_model = kwargs.pop('embedding_model', None)
        self.textblock = kwargs.pop('textblock', None)
        super().__init__(iterable, **kwargs)
        self.embedding = None

    def __call__(self):
        # generic (hopefully) call to embedding function - needs invoke function in
        # embedding_model to work
        self.embedding = self.embedding_model.invoke(self.textblock)
        return self.embedding
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    
    # Write a generic validator for the textblock. This needs to be hard coded
    # but maybe somewhere else - hopefully it can take in a format and validate
    # that format.
    def validate_textblock(self, value):
        pass

    # Add any extractors
    # example, if you want to extract code from a textblock, write a function to do that
    # here. Same if you wanted to extract out questions from a textblock. These 
    # extractors likely need to be hardcoded as functions. 
    # 

    
    # If the textblock
    # has a format specification that is passed to the validator, that should be
    # used to extract the relevant information here by specifying which kind of 
    # extractor to use for which aspect of the textblock.
        

