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


'''
Idea: 

You sample an LLM for a random state ({x=...})
get random action1 --> env(...) --> next_state1 --> advantage1 --> reset env() 
get random action2 --> env(...) --> next_state2 --> advantage2 --> reset env() 

==> DPO row 1: strategy prompt       action1       action2     (chosen rejected via higher advantage)

Note: use pre-trained value function to calculate advantage for next_state1, next_state2 --> reset env

FOR DPO, the dataset needs to be in this format: 
=========
Prompt:        
------
This is strategy prompt for LactChain --> we are tuning for [sequence of actions]

Chosen: 
-------
[move forward, move backward, ...]

Rejected: 
--------
[move left, move right, ...]

Note: 
----
Value Function has to be trained normally, and trained before we generate DPO dataset
That way it is good at generating advantages for states and actions

Intuition: 
---------
We use advantage to bootstrap DPO chosen and rejected because....
Gridworld returns (0) reward (negative reward in this case) up until you get to the goal, where u get a large positive reward. We need the advantage to estimate the "Return" so we can smooth out that signal across time so we do not just end up with a ton of negative rewards that mean nothing so our DPO markings are actually accurate. 
'''