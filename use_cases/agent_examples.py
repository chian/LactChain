import torch
import torch.nn as nn
import torch.optim as optim

class PolicyOnlyAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.01):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def policy(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        return self.policy(state)

    def update_policy(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.01):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.actor(state)
        return action_probs

    def evaluate(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        value = self.critic(state)
        return value

    def update(self, actor_loss, critic_loss):
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()