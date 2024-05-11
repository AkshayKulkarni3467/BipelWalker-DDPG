import copy
import gym
import torch
import random
import functools

import numpy as np
import torch.nn.functional as F

from collections import deque, namedtuple
from IPython.display import HTML
from base64 import b64encode

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from pytorch_lightning import LightningModule, Trainer
from gym.wrappers import RecordVideo, RecordEpisodeStatistics,Monitor
from pytorch_lightning.callbacks import ModelCheckpoint




device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()

def create_environment(env_name):
  env = gym.make(env_name)
  env = RecordVideo(env, video_folder='./videos2', episode_trigger=lambda x: x % 50 == 0)
  return env

class GradientPolicy(nn.Module):

  def __init__(self, hidden_size, obs_size, out_dims, min, max):
    super().__init__()
    self.min = torch.from_numpy(min).to(device)
    self.max = torch.from_numpy(max).to(device)
    self.net = nn.Sequential(
        nn.Linear(obs_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_dims),
        nn.Tanh()
    )
    
  def mu(self, x):
    if isinstance(x, np.ndarray):
      x = torch.from_numpy(x).to(device)
    return self.net(x.float()) * self.max

  def forward(self, x, epsilon=0):
    mu = self.mu(x)
    mu = mu + torch.normal(0, epsilon, mu.size(), device=mu.device)
    action = torch.max(torch.min(mu, self.max), self.min)
    action = action.cpu().numpy()
    return action

class DQN(nn.Module):

  def __init__(self, hidden_size, obs_size, out_dims):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(obs_size + out_dims, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),           
        nn.Linear(hidden_size, 1),
    )

  def forward(self, state, action):
    if isinstance(state, np.ndarray):
      state = torch.from_numpy(state).to(device)
    if isinstance(action, np.ndarray):
      action = torch.from_numpy(action).to(device)
    in_vector = torch.hstack((state, action))
    return self.net(in_vector.float())

class ReplayBuffer:

  def __init__(self, capacity):
    self.buffer = deque(maxlen=capacity)

  def __len__(self):
    return len(self.buffer)
  
  def append(self, experience):
    self.buffer.append(experience)
  
  def sample(self, batch_size):
    return random.sample(self.buffer, batch_size)

class RLDataset(IterableDataset):

  def __init__(self, buffer, sample_size=400):
    self.buffer = buffer
    self.sample_size = sample_size
  
  def __iter__(self):
    for experience in self.buffer.sample(self.sample_size):
      yield experience
      
def polyak_average(net, target_net, tau=0.01):
    for qp, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)
        
class DDPG(LightningModule):

  def __init__(self, env_name,capacity=100_000, 
               batch_size=256, actor_lr=1e-4, 
               critic_lr=1e-4, hidden_size=512, gamma=0.99, loss_fn=F.smooth_l1_loss, 
               optim=AdamW, eps_start=2.0, eps_end=0.2, eps_last_episode=1_000, samples_per_epoch=1_000, tau=0.005):

    super().__init__()

    self.env = create_environment(env_name)

    obs_size = self.env.observation_space.shape[0]
    action_dims = self.env.action_space.shape[0]
    max_action = self.env.action_space.high
    min_action = self.env.action_space.low
    self.batch_size = batch_size
    self.actor_lr = actor_lr
    self.critic_lr =critic_lr
    self.gamma = gamma
    self.loss_fn = loss_fn
    self.optim = optim
    self.eps_start = eps_start
    self.eps_end = eps_end
    self.eps_last_episode = eps_last_episode
    self.samples_per_epoch = samples_per_epoch
    self.tau = tau
    self.i = 0

    self.q_net = DQN(hidden_size, obs_size, action_dims)
    self.policy = GradientPolicy(hidden_size, obs_size, action_dims, min_action, max_action)

    self.target_policy = copy.deepcopy(self.policy)
    self.target_q_net = copy.deepcopy(self.q_net)

    self.buffer = ReplayBuffer(capacity=capacity)

    self.save_hyperparameters()

    while len(self.buffer) < self.samples_per_epoch:
      print(f"{len(self.buffer)} samples in experience buffer. Filling...")
      self.play_episode(epsilon=self.eps_start)

  @torch.no_grad()
  def play_episode(self, policy=None, epsilon=0.):
    obs = self.env.reset()
    done = False
    r = 0
    while not done:
      if policy:
        action = policy(obs, epsilon=epsilon)
      else:
        action = self.env.action_space.sample()
      self.env.render(mode='human')
      next_obs, reward, done, info = self.env.step(action)
      r = r+reward
      self.i = self.i+1
      exp = (obs, action, reward, done, next_obs)
      if self.i % 10 == 0:
        self.buffer.append(exp)
      obs = next_obs
    return r 

  def forward(self, x):
    output = self.policy(x)
    return output

  def configure_optimizers(self):
    q_net_optimizer = self.optim(self.q_net.parameters(), lr=self.critic_lr)
    policy_optimizer = self.optim(self.policy.parameters(), lr=self.actor_lr)
    return [q_net_optimizer, policy_optimizer]

  def train_dataloader(self):
    dataset = RLDataset(self.buffer, self.samples_per_epoch)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=self.batch_size,
    )
    return dataloader

  def training_step(self, batch, batch_idx, optimizer_idx):
    
    
    states, actions, rewards, dones, next_states = batch
    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1).bool()

    

    if optimizer_idx == 0:
      state_action_values = self.q_net(states, actions)
      next_state_values = self.target_q_net(next_states, self.target_policy.mu(next_states))
      next_state_values[dones] = 0.0
      expected_state_action_values = rewards + self.gamma * next_state_values
      q_loss = self.loss_fn(state_action_values, expected_state_action_values)
      self.log_dict({"episode/Q-Loss": q_loss})
      return q_loss
    
    elif optimizer_idx == 1:
      mu = self.policy.mu(states)
      policy_loss = - self.q_net(states, mu).mean()
      self.log_dict({"episode/Policy Loss": policy_loss})
      return policy_loss
  
  def training_epoch_end(self, outputs):
    epsilon = max(
        self.eps_end,
        self.eps_start - self.current_epoch / self.eps_last_episode
    )

    mean_reward = self.play_episode(policy=self.policy, epsilon=epsilon)

    polyak_average(self.q_net, self.target_q_net, tau=self.tau)
    polyak_average(self.policy, self.target_policy, tau=self.tau)
    
    self.log("episode/mean_reward", mean_reward)
    


    
    
algo = DDPG('BipedalWalker-v3')

checkpoint_callback = ModelCheckpoint(
    dirpath='models',
    filename = '{epoch}',
    every_n_epochs = 100
)

trainer = Trainer(checkpoint_callback=checkpoint_callback)

trainer = Trainer(
    gpus=num_gpus, 
    max_epochs=np.inf,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback]
)

trainer.fit(algo)