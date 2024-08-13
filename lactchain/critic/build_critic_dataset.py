from __future__ import annotations

'''Builds dataset for critic'''

from typing import Any, Literal
import gymnasium as gym
from pydantic import Field
from tqdm import tqdm
import sys, math

# imported configs
from lactchain.configs import BaseConfig
from lactchain.critic.buffer import CriticBufferConfig
from lactchain.environments.grid_world import GridWorldConfig
from lactchain.actor.actor_chain import ActorChainConfig
from lactchain.actor.strategy_chain import StrategyChainConfig
from lactchain.generators import VLLMGeneratorConfig
from lactchain.prompts import PromptTemplateConfigs

# import other lactchain packages
from lactchain.actor.actor_chain import ActorChain
from lactchain.actor.strategy_chain import StrategyChain
from lactchain.critic.buffer import CriticBuffer, CriticBufferConfig

from lactchain.prompts.actions import ActionsPromptTemplateConfig
from lactchain.prompts.strategies import StrategyPromptTemplateConfig

class SampleConfigs(BaseConfig):
    '''Joint Config for building up critic dataset'''

    name: Literal['sampling'] = 'sampling'

    batch_size: int = Field(
        default=32,
        description='Batch Size for sampling environment'
    )

    dataset_size: int = Field(
        default=100000,
        description='Desired dataset size for buffer'
    )

    gridworld_config: GridWorldConfig = Field(
        default_factory=GridWorldConfig
    )

    critic_buffer: CriticBufferConfig = Field(
        default_factory=CriticBufferConfig
    )

    generator_config: VLLMGeneratorConfig = Field(
        default_factory=VLLMGeneratorConfig
    )

    actor_chain_config: ActorChainConfig = Field(
        default_factory=ActorChainConfig
    )
    
    action_prompt_template_config: ActionsPromptTemplateConfig = Field(
        default_factory=ActionsPromptTemplateConfig
    )

    strategy_chain_config: StrategyChainConfig = Field(
        default_factory=StrategyChainConfig
    )
    
    strategy_prompt_template_config: StrategyPromptTemplateConfig = Field(
        default_factory=StrategyPromptTemplateConfig
    )


def initialize() -> dict[str, Any]:
    # import classes and environments
    from lactchain.actor.actor_chain import ActorChain
    from lactchain.actor.strategy_chain import StrategyChain
    from lactchain.environments import build_gridworld_env, process_environment_outputs
    from lactchain.generators import generator_factory
    from lactchain.prompts import get_prompt_template, get_action_solver

    config = SampleConfigs()
    
    # config.generator_config.pretrained_model_name_or_path = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
    # config.generator_config.pretrained_model_name_or_path = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa'
    # config.generator_config.pretrained_model_name_or_path = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/7129260dd854a80eb10ace5f61c20324b472b31c'
    # config.generator_config.pretrained_model_name_or_path = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--hugging-quants--Meta-Llama-3.1-70B-Instruct-AWQ-INT4/snapshots/e2017ac1487aa49826426f7d2382d596399beeb0'
    config.generator_config.pretrained_model_name_or_path = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--microsoft--Phi-3-medium-128k-instruct/snapshots/cae1d42b5577398fd1be9f0746052562ae552886'
    
    environment = gym.vector.AsyncVectorEnv([build_gridworld_env(config.gridworld_config)
                                             for _ in range(config.batch_size)])

    generator = generator_factory(config.generator_config)
    strategy_prompt_template = get_prompt_template(config.strategy_prompt_template_config)
    actor_prompt_template = get_prompt_template(config.action_prompt_template_config)
    solver = get_action_solver(config.action_prompt_template_config)
    
    strategy_chain = StrategyChain(generator=generator, prompt_template=strategy_prompt_template)
    actor_chain = ActorChain(generator, actor_prompt_template, solver, process_environment_outputs)
    buffer = CriticBuffer(config.critic_buffer)

    return {
        'environment': environment,
        'strategy_chain': strategy_chain,
        'actor_chain': actor_chain,
        'buffer': buffer
    }
    
    
def generate_data(environment: gym.Env, 
                  strategy_chain: StrategyChain, 
                  actor_chain: ActorChain, 
                  buffer: CriticBuffer
                  ): 
    
    BATCH_SIZE = 4
    MAX_BUFFER_SIZE = 1000
    
    observation, info = environment.reset() 
    environment_or_task = ['gridworld'] * BATCH_SIZE
    
    number_collection_steps = math.ceil(MAX_BUFFER_SIZE / BATCH_SIZE)
    
    progress_bar = tqdm(
        range(0, number_collection_steps),
        initial=0,
        leave=False,
        desc=f"COLLECTING DATA FOR LOCAL BUFFER OF SIZE {MAX_BUFFER_SIZE}",
        file=sys.stdout
    )
    
    while len(buffer) <= MAX_BUFFER_SIZE: 
        try:
            strategy = strategy_chain.sample_strategies(environment_or_task=environment_or_task)
            print(f'strategies: {strategy}')
            actions = actor_chain.sample_actions(strategy, observation, info)
            next_observation, reward, done, truncated, info = environment.step(actions)
            
            buffer.add_batch(rewards=reward, 
                            observations=observation, 
                            infos=info)
            
            observation = next_observation
            progress_bar.update(1)
            
        except Exception as e: 
            print(f'Exception in parsing: {e}, dropping batch...')
            continue
        
    return buffer


def main(): 
    '''Used for building critic dataset'''
    
    classes = initialize()
    buffer = generate_data(**classes)
    
    
    
if __name__=="__main__": 
    
    main()