# LactChain: Language Action Chain Reinforcement Learning

This repo serves as a template for coding out a Reinforcement Learning (RL) system. This system is meant to be a multi-purpose system with multiple possible applications.

## Installing the Environment

```
# if you are on polaris, make sure to activate
# anaconda modules via this command: 
https_proxy=http://proxy.alcf.anl.gov:3128
http_proxy=http://proxy.alcf.anl.gov:3128
module use /soft/modulefiles/
module load conda

# make sure you are in base directory 
# conda create -n lactchain python=3.11 -y
# saving into a project directory is preferred
conda create -p ../conda_envs/lactchain python=3.11 -y
#conda activate lactchain
conda activate ../conda_envs/lactchain 
pip install -e .
```

# Install Instructions: 
## Download Weights Via Cli
```
# download actor model
huggingface-cli download --repo-type model --cache-dir <your_directory_path> mistralai/Mistral-7B-Instruct-v0.3 --revision 83e9aa141f2e28c82232fea5325f54edf17c43de 

# download critic model
huggingface-cli download --repo-type model --cache-dir <your_directory_path> Salesforce/SFR-Embedding-Mistral
```

# Go To Working Directory: 
```
# cd to train folder 
cd lactchain/train
```

# Scripts
```
# If you are running from first time, these are in the lactchain/train/job_scripts/vllm subfolder 
# Note: Might require some hyperparam tuning
qsub vllm_server_small_train.sh

# If you are running from pretrained checkpoint
qsub ckpt_run_all_gpus.pbs
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

@startuml
actor User

' Participants
participant "User Code" as UC
participant ARGO_LLM
participant ArgoWrapper
participant ArgoEmbeddingWrapper
participant "Argo API"
participant ARGO_EMBEDDING
participant State
participant InputDict
participant "Embedding Model"
participant LactChain
participant AbstractEnvironment
participant GymEnv

note over User,UC: User interacts with the system

== ARGO_LLM Interaction ==

User -> UC: Instantiate ARGO_LLM
UC -> ARGO_LLM: __init__()
activate ARGO_LLM

note right of ARGO_LLM: Initializes with default parameters

UC -> ARGO_LLM: _call(prompt)
activate ARGO_LLM

ARGO_LLM -> ARGO_LLM: _get_model_default_parameters()
ARGO_LLM -> ArgoWrapper: invoke(prompt)
activate ArgoWrapper

ArgoWrapper -> ArgoWrapper: Prepare headers and data
ArgoWrapper -> json: dumps(data)
ArgoWrapper -> requests: post(url, headers, data_json)
activate requests

requests -> "Argo API": POST /chat/ with data
activate "Argo API"

"Argo API" --> requests: Response
deactivate "Argo API"

requests --> ArgoWrapper: response
deactivate requests

ArgoWrapper -> json: loads(response.text)
ArgoWrapper -> ARGO_LLM: parsed response
deactivate ArgoWrapper

ARGO_LLM -> ARGO_LLM: Extract response
ARGO_LLM --> UC: Return generated text
deactivate ARGO_LLM

== ARGO_EMBEDDING Interaction ==

UC -> ARGO_EMBEDDING: __init__(argo_wrapper)
activate ARGO_EMBEDDING

UC -> ARGO_EMBEDDING: embed_query(query)
activate ARGO_EMBEDDING

ARGO_EMBEDDING -> ARGO_EMBEDDING: _call(query)
ARGO_EMBEDDING -> ArgoEmbeddingWrapper: invoke(query)
activate ArgoEmbeddingWrapper

ArgoEmbeddingWrapper -> ArgoEmbeddingWrapper: Prepare headers and data
ArgoEmbeddingWrapper -> json: dumps(data)
ArgoEmbeddingWrapper -> requests: post(url, headers, data_json)
activate requests

requests -> "Argo API": POST /embed/ with data
activate "Argo API"

"Argo API" --> requests: Response
deactivate "Argo API"

requests --> ArgoEmbeddingWrapper: response
deactivate requests

ArgoEmbeddingWrapper -> json: loads(response.text)
ArgoEmbeddingWrapper --> ARGO_EMBEDDING: parsed response
deactivate ArgoEmbeddingWrapper

ARGO_EMBEDDING --> UC: Return embedding
deactivate ARGO_EMBEDDING

== State and LactChain Interaction ==

UC -> State: Instantiate State(embedding_model, textblock)
activate State

State -> InputDict: __init__()
activate InputDict
InputDict --> State: Initialize dictionary
deactivate InputDict

State -> State: __call__()
activate State

State -> "Embedding Model": invoke(textblock)
activate "Embedding Model"
"Embedding Model" --> State: embedding
deactivate "Embedding Model"

State --> UC: embedding
deactivate State

UC -> LactChain: Instantiate LactChain(state)
activate LactChain

LactChain -> State: Store state
deactivate LactChain

UC -> LactChain: transition(action)
activate LactChain

LactChain -> LactChain: Apply action to state
LactChain -> State: Update state
deactivate LactChain

== AbstractEnvironment Interaction ==

UC -> AbstractEnvironment: Instantiate
activate AbstractEnvironment
AbstractEnvironment -> GymEnv: __init__()
deactivate AbstractEnvironment

UC -> AbstractEnvironment: reset()
activate AbstractEnvironment
AbstractEnvironment --> UC: initial observation
deactivate AbstractEnvironment

UC -> AbstractEnvironment: step(action)
activate AbstractEnvironment
AbstractEnvironment -> LactChain: Apply action
AbstractEnvironment -> AbstractEnvironment: Compute reward
AbstractEnvironment --> UC: observation, reward, done, info
deactivate AbstractEnvironment

@enduml

