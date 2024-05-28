from pydantic import BaseModel, Field

class VisionConfig(BaseModel):
    in_channels:int=Field(4)
    out_channels:int=Field(10)
    kernel_size:int=Field(3)
    stride:int=Field(1)
    out_features:int=Field(128)
    dropout:float=Field(0.1)

class LinearConfig(BaseModel): 
    in_features:int=Field(4)
    out_features:int=Field(128)

class ActorConfig(BaseModel): 
    output_dim:int=Field(128)
    action_dim:int=Field(6)

class CriticConfig(BaseModel): 
    critic_dim:int=Field(1)
    gamma:float=Field(0.99)

class ACConfig(BaseModel): 
    game:str=Field('vision')
    model:BaseModel=Field(default_factory=VisionConfig)
    actor:ActorConfig=Field(default_factory=ActorConfig)
    critic:CriticConfig=Field(default_factory=CriticConfig)