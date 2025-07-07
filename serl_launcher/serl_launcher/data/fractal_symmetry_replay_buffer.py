import copy
from typing import Iterable, Optional, Tuple

import gym
import numpy as np
from serl_launcher.data.dataset import DatasetDict, _sample
from serl_launcher.data.replay_buffer import ReplayBuffer
from flax.core import frozen_dict
from gym.spaces import Box

class FractalSymmetryReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        split_method: str,
        split_freq: int, # For time-based only
        split_qs: int,
        branch_method: str,
    ):
        self.split_method=split_method
        self.split_qs=split_qs
        self.split_freq=split_freq
        self.branch_method=branch_method
        # TODO
        #   Initialize passed-in variables as needed 
        #   Add flags for variables passed to
        #   Function definition dependent on flags?

        super().__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity
        )

    # NOT TESTED
    def transform(self, data_dict: DatasetDict, translation: np.array):
        if not isinstance(data_dict, dict):
            return
        for k in data_dict.keys():
            if (self.placeholder): # if key is included in list provided
                data_dict[k] += translation
            else:
                self.transform(data_dict[k], translation)
    
    def fractal_branch(self, data_dict: DatasetDict):
        return
    
    def constant_branch(self, data_dict: DatasetDict):
        return
    
    def linear_branch(self, data_dict: DatasetDict):
        return
    
    def insert(self, data_dict: DatasetDict):
        return