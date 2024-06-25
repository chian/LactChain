from typing import List, Callable, Dict, Any, Tuple
from torch import Tensor
import torch
import gymnasium as gym
from datasets import Dataset as HFDataset
from argparse import ArgumentParser
from itertools import count
from lactchain.environments.grid_world import GridEnvironment
from lactchain.models.critic import ValueFunction, ValueFunctionConfig, LoraConfigSettings
from lactchain.models.actor import LactChain, ActorConfig, Strategy

def sample_data(env:gym.Env, num_samples:int) -> Tuple[Tensor, list[str]]:
    '''Samples a set of states and information strings used to compile strategy
    states: {'x':__, 'y':__, 'orientations':__}
    infos: 'Grid world size is __, you are at state __'
    '''
    distro_coord_space=env.coordinate_space_distro
    sampled_coords=distro_coord_space.sample((num_samples, 2))
    distro_orientation_space=env.orientation_set_distro
    sampled_orientations=distro_orientation_space.sample((num_samples,))
    
    infos=[env.info]*num_samples
    sampled_states=torch.cat([sampled_coords, sampled_orientations.unsqueeze(1)], dim=1)
    
    return sampled_states, infos

def batch_sample_states(sampled_states:Tensor, infos:list[str], batch_size:int) -> Tensor:
    '''Returns a batch of sampled_states
    Inputs:
    =======
    sampled_states: 2-dim tensor of X and Y integers used for compiling prompts as batches
    infos: list of informations used for compiling prompts as batches
    
    Outputs:
    =======
    states: list[Dict[str, int]] of len B
    infos: list[str] of len B 
    '''
    rand_batch_indices=torch.randint(0, sampled_states.size(0), (batch_size,))
    sampled_state_tensor=sampled_states[rand_batch_indices]

    states=[{'x':x, 'y':y, 'orientation':orientation} for (x, y, orientation) in sampled_state_tensor]

    return states, [infos[info] for info in rand_batch_indices]

def batch_sample_trajectories(states:list[Dict[str, int]], 
                              infos:list[int],
                              actor:LactChain, 
                              critic:ValueFunction, 
                              env:gym.Env
                              ) -> ...:
    '''
    
    
    '''
    actions, _=actor.sample_actions(states, infos) # batch B, but each two rows are same

    states_critic=[]
    infos_critic=[]
    rewards=[]
    num_skipped=0
    '''TODO: USE PROCESS POOL EXECUTOR TO AVOID SEQUENTIAL ENV RESET '''
    print(f'INTERACTING WITH ENVIRONMENT')
    for idx_1, in zip(count(0, step=1)):
        idx_2 = idx_1 + 1
        try:
            print(f'SAMPLING WITH BATCH SIZE {len(states)}')
            action1, action2=actions[idx_1], actions[idx_2]
            state1, reward1, _, info1=env.step(action1)
            _, _ = env.reset() # For now I reset env due to sequential passes
            state2, reward2, _, info2=env.step(action2)

            print(f'APPENDING STATES {idx_1} and {idx_2}')

            states_critic.append(state1)
            states_critic.append(state2)
            infos_critic.append(info1)
            infos_critic.append(info2)
            rewards.append(reward1)
            rewards.append(reward2)
        except Exception as e:
            print(f'ACTIONS NOT VALIDLY PARSED...SKIPPING SEQ...')
            num_skipped+=1
            pass

        if idx_2==len(actions)-1:
            print(f'FINISHED PROCESSING BATCH, KEPT {len(rewards)}')
            break
    print(f'CALCULATING ADVANTAGES FOR {len(rewards)}...')
    values=critic(states_critic, infos_critic)
    advantages=torch.tensor(rewards) - values
    evens, odds = (range(0, advantages.size(0)), 2), (range(1, advantages.size(0)), 2)

    prompts=[]
    chosens=[]
    rejects=[]
    print(f'FORMATTING INTO DATASET...')
    for even, odd, state, info in zip(evens, odds, states, infos):

        advantage1, advantage2=advantages[even], advantages[odd]
        print(f'Advantages: {advantage1} and {advantage2}')
        if advantage1 > advantage2:
            prompts.append(actor.compile_prompt(states[even], infos[even]))
            chosens.append(str(advantage1))
            rejects.append(str(advantage2))

        if advantage1 < advantage2:
            prompts.append(actor.compile_prompt(states[even], infos[even]))
            chosens.append(str(advantage2))
            rejects.append(str(advantage1))

        else:
            print(f'Advantages are the same, skipping...')
            num_skipped+=1
            pass

    return {'prompt':prompts, 'chosen':chosens,'rejected':rejects}

def sample_one_shot_trajectories(actor:LactChain=None,
                                 critic:ValueFunction=None,
                                 env:gym.Env=None,
                                 sampled_states:Tensor=None,
                                 infos:list[str]=None,
                                 num_samples:int=None
                                 ):
    print(f'GENERATING DPO DATASET...')
    rand_indices=torch.randint(0, sampled_states.size(0), (num_samples,))
    sampled_state_tensor=sampled_states[rand_indices]
    num_skipped=0
    prompts=[]
    chosens=[]
    rejects=[]

    for (x, y, orientation), info in zip(sampled_state_tensor, infos):
        state={'x':x, 'y':y, 'orientation':orientation}
        try:
            actions, _=actor.sample_actions([state]*2,[info]*2)
            out1, reward1, _, info1=env.step(actions[0])
            _, _ = env.reset()
            out2, reward2, _, info2=env.step(actions[1])

            values=critic([out1, out2], [info1, info2])
            advantages=torch.tensor([reward1, reward2]) - values
            print(f'Advantages: {advantages}')
            advantage_idx=torch.where(advantages>advantages.min().item())[0].item()

            print(f'ADDING ACTION TO DATASET...DATASET SIZE: {len(prompts+1)}')
            prompts.append(actor.compile_prompt(state, info))
            chosens.append(str(actions[advantage_idx]))
            rejects.append(str(actions[advantage_idx]))

        except Exception as e:
            print(f'ACTIONS NOT VALIDLY PARSED...SKIPPING SEQ...')
            num_skipped+=1
            pass

    print(f'NUMBER SEQ ACTIONS SKIPPED: {num_skipped}')

    return {'prompt':prompts, 'chosen':chosens,'rejected':rejects}

def argparse() -> Any:
    argparse=ArgumentParser()
    argparse.add_argument('--add_to_data', type=bool, default=True)
    argparse.add_argument('--data_save_path', type=str, default='./datasets/dataset_2')
    argparse.add_argument('--actor_path', type=str,
                          default='./models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de')
    argparse.add_argument('--critic_path', type=str,
                          default='./models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11')
    argparse.add_argument('--num_samples', type=int, default=100)
    argparse.add_argument('--batch_size', type=int, default=16)
    args = argparse.parse_args()
    return args

def main():

    args=argparse()
    env=GridEnvironment()
    actor_config=ActorConfig()
    critic_config=ValueFunctionConfig()
    lora_config=LoraConfigSettings()

    actor=LactChain(args.actor_path, actor_config, lora_config)
    critic=ValueFunction(args.critic_path, critic_config)

    sampled_states, infos=sample_data(env=env, num_samples=args.num_samples)
    # batch_samples, batch_infos=batch_sample_states(sampled_states, infos, args.batch_size)
    # data=batch_sample_actions(batch_samples, batch_infos, actor, critic, env)
    # breakpoint()

    data=sample_one_shot_trajectories(actor=actor, critic=critic, env=env,
                                      sampled_states=sampled_states, infos=infos,
                                      num_samples=args.num_samples)

    if args.data_save_path:
        
        if args.save
        
        
        dataset=HFDataset.from_dict(data)
        dataset.save_to_disk(args.data_save_path)
        print(f'Data saved at {args.data_save_path}')
        
        
if __name__=="__main__": 
    
    main()
