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


        self.experience = [[torch.tensor(x, dtype=torch.float) for x in transition] for transition in tqdm(zip(self.obs, self.action, self.reward, self.next_obs, self.terminated), desc='building dataset')]

    def __len__(self):
        return len(self.experience)
    
    def __getitem__(self, idx):
        return self.experience[idx]