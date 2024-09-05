from __future__ import annotations

'''Builds dataset for critic'''

from typing import Any, Literal, Optional
import gymnasium as gym
import torch
from torch import Tensor
from pydantic import Field
from tqdm import tqdm
import sys
import math

# imported configs
from lactchain.configs import BaseConfig
from lactchain.critic.buffer import CriticBufferConfig
from lactchain.environments.grid_world import GridWorldConfig
from lactchain.actor.actor_chain import ActorChainConfig
from lactchain.actor.strategy_chain import StrategyChainConfig
from lactchain.generators import VLLMGeneratorConfig

# import other lactchain packages
from lactchain.actor.actor_chain import ActorChain
from lactchain.actor.strategy_chain import StrategyChain
from lactchain.critic.buffer import CriticBuffer, CriticBufferConfig
from lactchain.prompts.actions import ActionsPromptTemplateConfig
from lactchain.prompts.strategies import StrategyPromptTemplateConfig

from lactchain.environments import build_gridworld_env, process_environment_outputs
from lactchain.generators import generator_factory
from lactchain.prompts import get_prompt_template, get_action_solver


class SampleConfigs(BaseConfig):
    '''Joint Config for building up critic dataset'''

    name: Literal['sampling'] = 'sampling'
    
    pretrained_model_name_or_path: str = Field(
        default='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de', 
        description='model path for generator'
    )

    batch_size: int = Field(
        default=256,
        description='Batch Size for sampling environment'
    )

    dataset_size: int = Field(
        default=10000,
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


def generate_data(batch_size: int,
                  dataset_size: int,
                  environment: gym.Env,
                  strategy_chain: StrategyChain,
                  actor_chain: ActorChain,
                  buffer: CriticBuffer
                  ) -> CriticBuffer:

    observation, info = environment.reset()
    environment_or_task = ['gridworld'] * batch_size
    number_collection_steps = math.ceil(dataset_size / batch_size)

    progress_bar = tqdm(
        range(0, number_collection_steps),
        initial=0,
        leave=False,
        desc=f"COLLECTING DATA FOR LOCAL BUFFER OF SIZE {dataset_size}",
        file=sys.stdout
    )

    while len(buffer) <= dataset_size:
        try:
            strategy = strategy_chain.sample_strategies(
                environment_or_task=environment_or_task)
            actions = actor_chain.sample_actions(strategy, observation, info)

            next_observation, reward, done, truncated, info = environment.step(
                actions)

            buffer.add_batch(rewards=reward,
                             observations=observation,
                             infos=info
                             )

            observation = next_observation
            progress_bar.update(1)

        except Exception as e:
            print(f'Exception in parsing: {e}, dropping batch...')
            continue

    return buffer


def main():
    '''Used for building critic dataset'''
    from datasets import Dataset as HFDataset
    from torch.utils.data import DataLoader

    from lactchain.critic.writer import Writer, HuggingFaceWriterConfig
    from lactchain.critic.critic_dataset import CriticDataset
    from lactchain.critic.tokenizer import CriticTokenizer, CriticTokenizerConfig
    from lactchain.critic.embedder import EmbedderFunction, EmbedderFunctionConfig

    config = SampleConfigs()
    
    environment = gym.vector.AsyncVectorEnv([build_gridworld_env(config.gridworld_config)
                                             for _ in range(config.batch_size)])

    generator = generator_factory(
        config.pretrained_model_name_or_path, config.generator_config)
    
    strategy_prompt_template = get_prompt_template(
        config.strategy_prompt_template_config)
    
    actor_prompt_template = get_prompt_template(
        config.action_prompt_template_config)
    
    solver = get_action_solver(
        config.action_prompt_template_config)

    strategy_chain = StrategyChain(
        generator=generator, prompt_template=strategy_prompt_template)
    
    actor_chain = ActorChain(
        generator, actor_prompt_template, solver, process_environment_outputs)
    
    buffer = CriticBuffer(config.critic_buffer,
                          process_fn=process_environment_outputs)
    
    writer_config = HuggingFaceWriterConfig()
    
    writer = Writer(writer_config)

    buffer = generate_data(config.batch_size, 
                           config.dataset_size, 
                           environment, 
                           strategy_chain, 
                           actor_chain, 
                           buffer)
    
    data = buffer.return_data()

    writer.write(output_dir='./critic_dataset', **data)


if __name__ == "__main__":

    main()
