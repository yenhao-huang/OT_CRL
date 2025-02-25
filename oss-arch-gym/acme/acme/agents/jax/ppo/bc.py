'''
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
'''

from copy import deepcopy
import tensorflow as tf
import pickle
import numpy as np
from pathlib import Path

IS_DEBUG = False
class BC(object):
    def __init__(self, fname, bc_cost, end_explore_updateidx):  
        self.first_round = False      
        self.expert_data = self._read_expert_data(fname)
        self.sample_size = 10000
        self.lr_rate = bc_cost
        self.end_explore_updateidx = end_explore_updateidx
    
    def _read_expert_data(self, fname):
        expert_data_path = Path(fname)
        if not expert_data_path.exists():
            self.first_round = True
            expert_data = None
        else:
            with expert_data_path.open("rb") as f:
                expert_data = pickle.load(f)

        return expert_data
    
    def _preprocess(self, state, scaler):
        state_norm = scaler.transform(state)
        return state_norm
    
    def sample(self):
        state, action, metadata = self.expert_data[0], self.expert_data[1], self.expert_data[2]
        if IS_DEBUG:
            print(metadata["len"])
            raise "1"
        if self.sample_size > metadata["len"]:
            state_norm = self._preprocess(state, metadata["scaler"])
            return state_norm, action
        else:
            idx = np.random.choice(list(range(metadata["len"])), size=self.sample_size, replace=False)
            state_norm = self._preprocess(state[idx], metadata["scaler"])
            return state_norm, action[idx]
