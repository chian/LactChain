'''
=========
Questions: 
=========

1.) For Env in GridWorld-LLM, input is sequence of actions ['move left', 'move forward', ...] 
--> Output is ({x:__, y:__, orientation:__}, reward=100, done=bool). 

Question: Is the input aka "button" a sequence of actions or one action?: 
A: button is a sequence of actions so sequence of actions is one action

Question: Is the output reward the summed reward for the sequence of actions, or a list of rewards per action?
A: it is supposed to be one reward 
Advantage = reward_t + gamma*Q(a_t+1, s_t+1) - Q(s_t, a_t)


2.) For LactChain, it is an LLM with 2 things:

Strategy Prompt (This is the thing that is learnable as the policy):
============== 
You are in gridworld. Make a move to help you reach the goal. 
Your response must be some kind of move, even if you have to guess. 

Strategy Conversion (This is fixed, and maps strategy prompt --> something discrete): 
===================
There are only 2 types of moves you can make:

1. move forward
2. turn left

Come up with a combination of those two moves in order
to successfully carry out the action: {strategy}

Your final answer should be in the format of a python list 
of moves, where each move is one of the 2 types listed above.
E.g. ['move forward', 'turn left']

Questions: 
- The Policy is two fold: what lactchain to use, and the strategy/content of that lactchain
- Policy is learned via DPO/SinPO on data {strategy win, strategy loss}

- Update to policy ~ policy_param + alpha * Q(a_t, s_t) * grad{log_prob(action from policy)}
==> Q value is from AutoModel, but how do we get log_prob of action?
A: Below  

For DPO: 
- What I thought (wrong) DPO 1: we send in state twice (x, y, orientation), and get two actions out {[move forward, turn left], 
[move forward, move forward, ...]}. Two actions are sent into the environment 


-  (This one is actual one) DPO 2: for episode, we send in action and get state and reward, and save it in a memory database. 
At the end of an episode, we sample two trajectories with 


Loop: 
- obs --> We add strategy + strategy converter --> (LLM1) --> button aka seq actions --> env 
--> one reward, next obs, next state --> save in memory --> repeat until end of episode 

At episode end: 
Sample two trajectories from memory, where the beginning state has to match
--> (state_0, action1, advantage or reward or some penalty signal) + (state_0, action2, advantage or reward or some penalty signal)
--> Use this in DPO, KTO

--> penalize LLM policy with DPO where the {1, 0} is the {action pref, action less}
--> and penalize value function with values stored in memory 
--> value function does not have to be llm

in general we want llm --> state representation --> make emb --> nn{...} --> Value 

- why cant we do contrastive style of action proposal?

The action is assigned W and L based on value function 
(higher value function is assigned W, lower value is assigned L) for DPO


'''