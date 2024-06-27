# LactChain: Language Action Chain Reinforcement Learning

This repo serves as a template for coding out a Reinforcement Learning (RL) system. This system is meant to be a multi-purpose system with multiple possible applications.

# Install Instructions: 
## Download weights: 
```
# download actor model
huggingface-cli download --repo-type models --cache-dir ./ mistralai/Mistral-7B-Instruct-v0.3
# download critic model
huggingface-cli download --repo-type models --cache-dir ./ Salesforce/SFR-Embedding-Mistral
```
## Note: Make sure you are on cuda devices 12

```
conda create -n lactchain python=3.11 -y
conda activate lactchain 
pip install -e .
```

## Current Components: 
Components: 
1.) Environment: 
Input: list of actions --> Output: Reward, Next State, Info (Note: Info is used to inform Lactchain)

- Takes in a list of actions from Lactchain 
- Maps left, right to number 0, 1 --> Run through grid 
- Collect reward for sequence of actions 
- State: {x, y, orientation}
- Info: 'Grid world is size 4x4'

2.) LactChain: 
Input: State + Info --> Output: Next action

- Mistral 7B-instruct V3 with LoRA 
- Has a prompt that comprises: 
    - Strategy: 'You are in grid world, propose sequence of moves [left, right]....'
    - State + info: '{x, y, z}, grid world is size...'

- Output is sequence of actions 

3.) ValueFunction: 
Input: State + Info --> Output: Q - Value 

- Mistral SFR + LoRA 
- Concat [state, info] --> (768,) --> Q-head --> Q value 1-dim

## Training Regime: 
1.) Training Value Function

Freeze Actor + Init Critic

Done with Monte Carlo Style: 

For i in num_episodes: 

    obs, info = env.reset()
    rewards=[]
    values=[]

    While not done or step <= max steps: 

        1.) Actor(obs, info) --> action
        2.) Env(action) --> reward, state, info 
        3.) Critic(state, info) --> Q-value 
        4.) rewards.append(reward) and values.append(value)

    @Episode end: 
    1.) Use rewards to calculate list of returns: 
        [R0 = r0, 
        R1 = r0 + gamma * r1, 
        ....
        ]
    2.) Calculate Advantages via list_returns - list_values 
    3.) Loss, backprop 


2.) Train SimPO: 

Init Actor + Freeze Trained Critic from 1

@DataCollection: 
    Create State Tensor Z for Sampling:
    1.) Create 2 Distros of coord space + orientation space via torch.distros + env
    2.) Sample tensor of size (num_samples) from distros

    3.) Sample one x, y, orientation and info from State Tensor Z, then make two copies [x, y, ...]*2
    4.) Pass into actor --> get two actions [action1, action2]

    5.) Send two actions [action1, action2] --> Env(...) --> [reward1, state1, info1] + [reward2, state2, info2] --> Then compute returnes [R1, R2] with rewards list

    6.) Send actions [action1, action2] into Critic --> [Q1, Q2]
    7.) Compute Advantages = [Q1, Q2] - [R1, R2] = [A1, A2]
    8.) Dataset: 
    ===========================================
    prompt          chosen             rejected
    [x, y,          [action1]          [action2]
    orientation, 
    info]

@Done Data Collection 
SIMPO Train 


## Questions / Concerns: 

- I am noticing that the agent never finishes grid world due to it keep running off the map. 

I can prompt tweak, but concerned if q only has small time 

- None of this is distributed 

Data Gather is extremely slow 

- Checkpointing 

How should I checkpoint this? Should I do this based on run? Or checkpoint over multiple runs?

## TODOs:

Add things needed to enforce structure of any subsequent code
1. Make generic baseclass for actor (policy) network
2. Make generic baseclass for critic (value) network
3. Make generic baseclass for reward function
4. Finish thinking about generic lactchain baseclass. Yes, it is state-->action-->state, but what is action? Does action involve taking in a fluid prompt? A prompt menu? What?
5. Write unit tests

Build out specific use cases
1. Draw schematic of simple use case
2. Add plausibly useful language action chains using lactchain class
3. Add code extractor and other functions in state class
4. Add other extractors to lactchains if you need to pull certain things (like code) from gpt4 responses
5. Define example format for textblock in state class
6. Define Policy and Value Function networks
7. Define Actor-Critic teaching moments (TD learning? Whatever it's called)



