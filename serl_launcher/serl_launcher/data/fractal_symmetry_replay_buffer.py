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
        branch_method: str,
        workspace_width: int,
        depth: int = 1,
        dendrites: int = None, 
        timesplit_freq: int = None,
    ):

        self.split_method=split_method # Determines method for when to change transformation number
        self.branch_method=branch_method # Determines method for how many transforms to make for a given transition
        self.workspace_width = workspace_width # Total 1D distance for transform calculations
        self.depth=depth # Total number of layers
        self.dendrites=dendrites # For fractal only
        self.timesplit_freq=timesplit_freq # For time-based only
        

        # TODO
        #   Initialize passed-in variables as needed 
        #   Add flags for variables passed to
        #   Add datastore class in datastore.py
        #   Create transform function
        #   Create at least one functional branch_method
        #   Create at least one functional split_method
        #   Create automated test.py with tests for current methods
        #   Create more methods with tests
        # 
        # Future considerations:
        #   get_pos_dict_paths to return a list of paths to alter for more complex environments
        #   when dealing with x and y for transforms, consider two versions:
        #       radial trees at different angles can have strong fractal symmetry built in but will be more difficult to evenly space lowest level transforms
        #       grid-based will inherently evenly space lowest transforms, but fractal symmetry will be restricted to 9 dendrites (original plus 8) in order to conform

        super().__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity
        )

    def transform(self, data_dict: DatasetDict, translation: np.array):
        # TBD:
        #   return data_dict with positional arguments += translation
        #   return data_dict with positional arguments = translation
        return
    
    def fractal_branch(self, data_dict: DatasetDict):
        # return a new number of branches = dendrites ^ depth
        return
    
    def constant_branch(self, data_dict: DatasetDict):
        # return current number of branches
        return
    
    def linear_branch(self, data_dict: DatasetDict):
        # return a new number of branches = branches_count + n
        return
    
    def time_split(self, data_dict: DatasetDict):
        # return True when a set time has passed
        return False
    
    def rel_pos_split(self, data_dict: DatasetDict):
        # return True if there is a change of depth determined by relative position
        return False
    
    def height_split(self, data_dict: DatasetDict):
        # return True if there is a change of depth determined by height
        return False
    
    def velocity_split(self, data_dict: DatasetDict):
        # return True if there is a change of depth determined by velocity
        return False
    
    def constant_split(self, data_dict: DatasetDict):
        # return True every transition
        return True
    
    def insert(self, data_dict: DatasetDict):
        match self.split_method:
            case "time":
                return
            case "rel_pos":
                return
            case "height":
                return
            case "vel":
                print("whoop-dee-doo")
            case _:
                return
            
        match self.branch_method:
            case "fractal":
                self.
            case "linear":
                return
            case "constant":
                return
            case _:
                return
            
        
