import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset
import gym
from tqdm import tqdm

from model import Actor, Critic
from noise_generator import GaussianNoise
from dataset import OfflineDataset
# from replay_buffer import ReplayMemory  
import os
from collections import namedtuple

Transition = namedtuple('Transition', ['obs', 'action', 'reward', 'next_obs', 'terminated'])

class PretrainedTD3():
    def __init__(self, obs_space, action_space, args, device):
        self.obs_dim = obs_space.shape
        self.action_dim = action_space.shape
        self.env_name = args.env_name
        self.dataset_size = args.dataset_size
        self.load_path = args.load_path
        self.device = device

        self.actor = Actor(self.obs_dim[0], self.action_dim[0]).to(device)
        self.actor.load_state_dict(torch.load(args.load_path))

        self.write_transitions = args.write_transitions
        self.transition_path = args.transition_path

    def generate_dataset(self, env):
        if not self.write_transitions and os.path.exists(self.transition_path):
            print(f'Reading existing data from {self.transition_path}')
            return OfflineDataset(self.transition_path)
            

        print(f'Writing new data... (read_transition is set to false or the transition_path does not exist )')
        seed = 0
        obs = env.reset(seed=seed)
        obs_records = []
        action_records = []
        reward_records = []
        next_obs_records = []
        terminated_records = []

        for _ in tqdm(range(self.dataset_size), desc=f'Generating data from {self.load_path}'):
            action = self.select_action(obs)
            next_obs, reward, terminated, _ = env.step(action)

            obs_records.append(obs)
            action_records.append(action)
            next_obs_records.append(next_obs)
            reward_records.append(reward)
            terminated_records.append(terminated)
            
            obs = next_obs

            if terminated:
                seed += 1
                obs = env.reset(seed=seed)

        if not os.path.exists(self.transition_path):
            os.mkdir(self.transition_path)
        
        np.save(os.path.join(self.transition_path, 'obs'), np.array(obs_records))
        np.save(os.path.join(self.transition_path, 'action'), np.array(action_records))
        np.save(os.path.join(self.transition_path, 'next_obs'), np.array(next_obs_records))
        np.save(os.path.join(self.transition_path, 'reward'), np.array(reward_records))
        np.save(os.path.join(self.transition_path, 'terminated'), np.array(terminated_records))

        return OfflineDataset(self.transition_path)

    def select_action(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device)
        action = self.actor(obs).cpu().detach().numpy()
        return action


class TD3Agent:
    def __init__(self, obs_space, action_space, args, device):
        self.obs_dim = obs_space.shape
        self.action_dim = action_space.shape

        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_low_tensor = torch.from_numpy(action_space.low).float().to(device)
        self.action_high_tensor = torch.from_numpy(action_space.high).float().to(device)


        self.critic1 = Critic(self.obs_dim[0], self.action_dim[0]).to(device)
        self.critic2 = Critic(self.obs_dim[0], self.action_dim[0]).to(device)
        self.target_critic1 = Critic(self.obs_dim[0], self.action_dim[0]).to(device)
        self.target_critic2 = Critic(self.obs_dim[0], self.action_dim[0]).to(device)

        self.actor = Actor(self.obs_dim[0], self.action_dim[0]).to(device)
        self.target_actor = Actor(self.obs_dim[0], self.action_dim[0]).to(device)

        self.hard_update(self.target_critic1, self.critic1)
        self.hard_update(self.target_critic2, self.critic2)
        self.hard_update(self.target_actor, self.actor)

        self.critic1_optim = optim.Adam(params=self.critic1.parameters(), lr=args.lr)
        self.critic2_optim = optim.Adam(params=self.critic2.parameters(), lr=args.lr)
        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=args.lr)
        self.mse_loss = nn.MSELoss()

        self.policy_smoother = GaussianNoise(size=self.action_dim, mu=args.smoother_mu, sigma=args.smoother_sigma, clip=args.smoother_clip)


        # self.memory = ReplayMemory(capacity=args.max_steps) # the entire history of the agent
        self.batch_size = args.batch_size
        self.device = device

        self.steps = 0 # for actor and target critic update
        self.eval_freq = args.eval_freq
        self.gamma = args.gamma
        self.delay = args.delay
        self.tau = args.tau
        self.env_name = args.env_name
        self.eval_episodes = args.eval_episodes


        self.save_dir = args.save_dir

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device)
            action = self.actor(obs).cpu().detach().numpy()

            return action

    
    def hard_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)


    def evaluate(self):
        env = gym.make(self.env_name)

        avg_return = 0.
        for i in range(self.eval_episodes):
            obs = env.reset()
            total_reward = 0
            while True:
                action = self.select_action(obs)
                obs, reward, terminated, _ = env.step(action)
                avg_return += reward
                total_reward += reward

                if terminated:
                    break
            print(f'total reward of  eval episode {i}: {total_reward}')

        avg_return /= self.eval_episodes

        return avg_return


    def learn(self, experience):
        experience = [x.to(self.device) for x in experience]
        obs, action, reward, next_obs, terminated = experience

        # unsqueeze to prevent incorrect broadcasting for mujoco env
        reward = reward.unsqueeze(dim=1)
        terminated = terminated.unsqueeze(dim=1)
        
        # build q target
        with torch.no_grad():
            next_action = self.target_actor(next_obs) 
            next_action += torch.from_numpy(self.policy_smoother.sample()).float().to(self.device)
            next_action = torch.clamp(next_action, min=self.action_low_tensor, max=self.action_high_tensor)

            q_target = reward + self.gamma * (1 - terminated) * torch.min(self.target_critic1(next_obs, next_action), self.target_critic2(next_obs, next_action))

        # update critic1
        q_pred1 = self.critic1(obs, action)
        critic_loss = self.mse_loss(q_pred1, q_target.detach())
        self.critic1_optim.zero_grad()
        critic_loss.backward()
        self.critic1_optim.step()

        # update critic2
        q_pred2 = self.critic2(obs, action)
        critic_loss = self.mse_loss(q_pred2, q_target.detach())
        self.critic2_optim.zero_grad()
        critic_loss.backward()
        self.critic2_optim.step()


        # update actor and target network
        if self.steps % self.delay == 0:

            # update actor
            grad = -self.critic1(obs, self.actor(obs)).mean() # negative sign for gradient ascend
            self.actor_optim.zero_grad()
            grad.backward()
            self.actor_optim.step()


            # soft update target network
            for source_param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

            for source_param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

            for source_param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)


        
    def save_model(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        torch.save(self.actor.state_dict(), f'{self.save_dir}/actor_{self.steps+1}.pth')
        torch.save(self.critic1.state_dict(), f'{self.save_dir}/critic1_{self.steps+1}.pth')
        torch.save(self.critic2.state_dict(), f'{self.save_dir}/critic2_{self.steps+1}.pth')

        