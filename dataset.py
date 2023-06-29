from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm

import os


class OfflineDataset(Dataset):
    def __init__(self, transition_path):
        if transition_path is not None:
            self.obs = np.load(os.path.join(transition_path, 'obs.npy'))
            self.action = np.load(os.path.join(transition_path, 'action.npy'))
            self.reward = np.load(os.path.join(transition_path, 'reward.npy'))
            self.next_obs = np.load(os.path.join(transition_path, 'next_obs.npy'))
            self.terminated = np.load(os.path.join(transition_path, 'terminated.npy'))


    def normalize_states(self, eps):
        mean = np.mean(self.obs, axis=0)
        std = np.std(self.obs, axis=0) + eps
        self.obs = (self.obs - mean) / std
        self.next_obs = (self.next_obs - mean) / std
        return mean, std

    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.obs[idx], dtype=torch.float), torch.tensor(self.action[idx], dtype=torch.float), torch.tensor(self.reward[idx], dtype=torch.float), torch.tensor(self.next_obs[idx], dtype=torch.float), torch.tensor(self.terminated[idx], dtype=torch.float)