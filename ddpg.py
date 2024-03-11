
"""
    An implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm
    for continuous control tasks in a deep learning.
"""

import os, torch, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import Actor, Critic
from torch.optim import AdamW
from collections import deque

try:
    import unzip_requirements
except ImportError:
    pass
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from actions import actions

class Agent:
    """
    Manages ddpg process, critic, actor and memory
    env: (class instance) extends gym.Env
    nsteps: (int) training steps to set noise decay
    gamma: (float) discount factor
    tau: (float) soft target update factor
    actor/critic params = [
        input (state) size,
        output (action) size,
        [hidden_layer_1, hidden_layer_2],
        init weight, init bias (need to match defaults for serving)
    ]
    lr = learning rate
    memory params = (buffer length, sample batch size)
    """
    def __init__(self, env, nsteps=None, gamma=.99, tau=.001, seed=None):
        self.gamma = gamma
        self.tau = tau
        self.seed = seed
        self.loss = [0, 0]
        if seed:
            torch.manual_seed(seed)

        nactions = env.action_space.shape[0]
        params = [env.observation_space.shape[0], nactions, [64, 48], .003, .0003]
        self.actor = Actor(*params)
        self.actor_target = Actor(*params)
        self.critic = Critic(*params)
        self.critic_target = Critic(*params)

        self.critic_criterion  = nn.MSELoss()
        self.actor_opt  = AdamW(self.actor.parameters())
        self.critic_opt = AdamW(self.critic.parameters())

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        if nsteps:
            mem = min(nsteps*100, 50000)
            self.memory = Memory(mem, 48)
            self.noise = Noise(nactions, nsteps)

    def reset(self):
        self.noise.reset()

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data*(1.0-self.tau) + param.data*self.tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def act(self, state):
        state = torch.from_numpy(state).float()
        a = self.actor.forward(state).detach().numpy()
        n = self.noise()
        action = np.add(a, n)
        action = action.clip(-1, 1)
        return action, a, n

    def update(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) > self.memory.batch:
            states, actions, rewards, next_states, _ = self.memory.sample()

            # Critic loss
            Qvals = self.critic.forward(states, actions)
            next_actions = self.actor_target.forward(next_states)
            next_Q = self.critic_target.forward(next_states, next_actions.detach())
            Qprime = rewards + self.gamma * next_Q
            critic_loss = self.critic_criterion(Qvals, Qprime)

            # Actor loss
            policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

            # update networks
            self.actor_opt.zero_grad()
            policy_loss.backward()
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward() 
            self.critic_opt.step()

            # update target networks
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)

            if policy_loss:
                self.loss[0] = policy_loss.detach().float().numpy()
            if critic_loss:
                self.loss[1] = critic_loss.detach().float().numpy()

    def save(self, fin, fout, reward=None, score=None):
        state = {
            'path': fin, 'reward': reward, 'score': score,
            'gamma': self.gamma, 'tau': self.tau, 'seed': self.seed,
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(), 
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_criterion': self.critic_criterion,
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'memory': self.memory,
            'noise': self.noise,
        }
        torch.save(state, fout, _use_new_zipfile_serialization=False)

    def load(self, fn):
        a = torch.load(fn)
        self.gamma = a['gamma']
        self.tau = a['tau']
        self.actor.load_state_dict(a['actor'])
        self.actor_target.load_state_dict(a['actor_target'])
        self.critic.load_state_dict(a['critic'])
        self.critic_target.load_state_dict(a['critic_target'])
        self.critic_criterion = a['critic_criterion']
        self.actor_opt.load_state_dict(a['actor_opt'])
        self.critic_opt.load_state_dict(a['critic_opt'])
        self.memory = a['memory']
        self.noise = a['noise']
        if a['seed']:
            torch.manual_seed(a['seed'])
        state = {'modelpath': a['path'], 'reward': a['reward'], 'score': a['score']}

class Memory:
    def __init__(self, max_size, batch):
        self.buffer = deque(maxlen=max_size)
        self.batch = batch

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self):
        batch = random.sample(self.buffer, self.batch)
        batch = [torch.FloatTensor(x) for x in zip(*batch)]
        return batch

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden=[64, 48], weight=.003, bias=.0003):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_size, hidden[0])
        self.layer2 = nn.Linear(hidden[0], hidden[1])
        self.output = nn.Linear(hidden[1], action_size)

        self.norm1 = nn.LayerNorm(hidden[0])
        self.norm2 = nn.LayerNorm(hidden[1])

        nn.init.uniform_(self.layer1.weight, 1./sqrt(hidden[0]))
        nn.init.uniform_(self.layer1.bias, 1./sqrt(hidden[0]))
        nn.init.uniform_(self.layer2.weight, 1./sqrt(hidden[1]))
        nn.init.uniform_(self.layer2.bias, 1./sqrt(hidden[1]))
        nn.init.uniform_(self.output.weight, -weight, weight)
        nn.init.uniform_(self.output.bias, -bias, bias)

    def forward(self, state):
        x = F.relu(self.norm1(self.layer1(state)))
        x = F.relu(self.norm2(self.layer2(x)))
        return torch.tanh(self.output(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden, weight, bias):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_size, hidden[0])
        self.layer2 = nn.Linear(hidden[0]+action_size, hidden[1])
        self.output = nn.Linear(hidden[1], 1)

        self.norm1 = nn.LayerNorm(hidden[0])
        self.norm2 = nn.LayerNorm(hidden[1])

        nn.init.uniform_(self.layer1.weight, 1./sqrt(hidden[0]))
        nn.init.uniform_(self.layer1.bias, 1./sqrt(hidden[0]))
        nn.init.uniform_(self.layer2.weight, 1./sqrt(hidden[1]))
        nn.init.uniform_(self.layer2.bias, 1./sqrt(hidden[1]))
        nn.init.uniform_(self.output.weight, -weight, weight)
        nn.init.uniform_(self.output.bias, -bias, bias)

    def forward(self, state, action):
        x = F.relu(self.norm1(self.layer1(state)))
        x = F.relu(self.norm2(self.layer2(torch.cat([x, action], 1))))
        return self.output(x)
