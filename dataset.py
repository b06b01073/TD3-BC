from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm

import os


class OfflineDataset(Dataset):
    def __init__(self, transition_path):
        ''' 
        Args:
            experience: is a list of namedtuple('Transition', ['obs', 'action', 'reward', 'next_obs', 'terminated'])
        '''
        if transition_path is not None:
            self.obs = np.load(os.path.join(transition_path, 'obs.npy'))
            self.action = np.load(os.path.join(transition_path, 'action.npy'))
            self.reward = np.load(os.path.join(transition_path, 'reward.npy'))
            self.next_obs = np.load(os.path.join(transition_path, 'next_obs.npy'))
            self.terminated = np.load(os.path.join(transition_path, 'terminated.npy'))

        # expand dims to prevent iteration over a 0-d array in the list comprehension when constructing self.experience
        # self.reward = np.expand_dims(self.reward, axis=1)
        # self.terminated = np.expand_dims(self.terminated, axis=1)

        self.experience = [[torch.tensor(x, dtype=torch.float) for x in transition] for transition in tqdm(zip(self.obs, self.action, self.reward, self.next_obs, self.terminated), desc='building dataset')]

    def __len__(self):
        return len(self.experience)
    
    def __getitem__(self, idx):
        return self.experience[idx]