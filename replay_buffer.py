import numpy as np
import torch

import os


class OfflineBuffer:
    def __init__(self, data_path):
        self.obs = np.load(os.path.join(data_path, 'obs.npy'))
        self.actions = np.load(os.path.join(data_path, 'action.npy'))
        self.rewards = np.load(os.path.join(data_path, 'reward.npy'))
        self.next_obs = np.load(os.path.join(data_path, 'next_obs.npy'))
        self.terminated = np.load(os.path.join(data_path, 'terminated.npy'))

        self.len = min(len(data) for data in [self.obs, self.actions, self.rewards, self.next_obs, self.terminated])


    def normalize_states(self, eps):
        mean = np.mean(self.obs, axis=0)
        std = np.std(self.obs, axis=0) + eps
        self.obs = (self.obs - mean) / std
        self.next_obs = (self.next_obs - mean) / std
        return mean, std

    
    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        if batch_size > self.len:
            return None
        
        indices = np.random.randint(low=0, high=self.len, size=batch_size)
        transitions = (self.obs[indices], self.actions[indices], self.rewards[indices], self.next_obs[indices], self.terminated[indices])

        return (torch.tensor(x, dtype=torch.float, device=device) for x in transitions)

