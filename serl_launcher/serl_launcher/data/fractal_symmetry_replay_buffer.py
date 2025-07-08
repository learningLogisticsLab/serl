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

        #Potentially temporary variables
        depth: int = 1,
        dendrites: int = None, 
        timesplit_freq: int = None,
        branch_count_rate_of_change: int = None,
    ):

        self.split_method=split_method # Determines method for when to change transformation number
        self.branch_method=branch_method # Determines method for how many transforms to make for a given transition
        self.workspace_width = workspace_width # Total 1D distance for transform calculations

        # Potentially temporary variables
        self.current_branch_count = 1 # Current number of branches
        self.current_depth = 1 # Current depth
        self.depth=depth # Total number of layers
        self.dendrites=dendrites # For fractal only
        self.timesplit_freq=timesplit_freq # For time-based only
        self.branch_count_rate_of_change=branch_count_rate_of_change # For linear only

        # TODO
        #   Initialize passed-in variables as needed
        #   Temporary variables that are optional should be part of a *args or **kwargs 
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

    # REQUIRES TESTING
    def transform(self, data_dict: DatasetDict, translation: np.array):
        #   return data_dict with positional arguments += translation
        data_dict["observations"]["state"] += translation
        data_dict["next_observations"]["state"] += translation
    
    # REQUIRES TESTING
    def fractal_branch(self, data_dict: DatasetDict):
        # return a new number of branches = dendrites ^ depth
        self.depth += 1
        return self.dendrites ** (self.depth - 1)
    
    # REQUIRES TESTING
    def constant_branch(self, data_dict: DatasetDict):
        # return current number of branches
        return self.current_branch_count
    
    # REQUIRES TESTING
    def linear_branch(self, data_dict: DatasetDict):
        # return a new number of branches = branches_count + n
        return self.current_branch_count + self.branch_count_rate_of_change
    
    # UNFINISHED
    def time_split(self, data_dict: DatasetDict):
        # return True when a set time has passed
        return False
    
    # UNFINISHED
    def rel_pos_split(self, data_dict: DatasetDict):
        # return True if there is a change of depth determined by relative position
        return False
    
    # UNFINISHED
    def height_split(self, data_dict: DatasetDict):
        # return True if there is a change of depth determined by height
        return False
    
    # UNFINISHED
    def velocity_split(self, data_dict: DatasetDict):
        # return True if there is a change of depth determined by velocity
        return False
    
    def constant_split(self, data_dict: DatasetDict):
        # return True every transition
        return True
    
    # REQUIRES TESTING
    def insert(self, data_dict: DatasetDict):
        # Choose when to change number of branches
        match self.split_method:
            case "time":
                split = self.time_split
            case "rel_pos":
                split = self.rel_pos_split
            case "height":
                split = self.height_split
            case "velocity":
                split = self.velocity_split
            case _:
                raise ValueError("incorrect value passed to split_method")
            
        # Choose how number of branches changes 
        match self.branch_method:
            case "fractal":
                branch = self.fractal_branch
            case "linear":
                branch = self.linear_branch
            case "constant":
                branch = self.constant_branch
            case _:
                raise ValueError("incorrect value passed to branch_method")
        
        # Update number of branches if needed
        if split(data_dict):
            self.current_branch_count = branch(data_dict)
        
        #---------------------TEMPORARY SECTION TO BE IMPROVED------------------------#
        # Save positions
        rx = data_dict["observations"]["state"][0]
        # ry = data_dict["observations"]["state"][1]
        bx = data_dict["observations"]["state"][7]
        # by = data_dict["observations"]["state"][8]

        rx2 = data_dict["next_observations"]["state"][0]
        # ry2 = data_dict["next_observations"]["state"][1]
        bx2 = data_dict["next_observations"]["state"][7]
        # by2 = data_dict["next_observations"]["state"][8]

        # OR set to extreme for iterations
        x = self.workspace_width / 2
        data_dict["observations"]["state"] -= np.array([x, 0, 0, 0, 0, 0, 0, x, 0, 0])
        data_dict["next_observations"]["state"] -= np.array([x, 0, 0, 0, 0, 0, 0, x, 0, 0])
        #-----------------------------------------------------------------------------#

        from_left = []
        from_center = []

        # Initial insert of most-extreme branch
        dx = self.workspace_width / (self.current_branch_count * 2)
        super().insert(self.transform(data_dict, np.array([dx, 0, 0, 0, 0, 0, 0, dx, 0, 0])))
        dx = dx * 2

        from_left.append(data_dict["observations"]["state"][0] - self.workspace_width)
        from_center.append(data_dict["observations"]["state"][0] - rx)

        # Insert of rest of branches
        for t in range(0, self.current_branch_count - 1):
            super().insert(self.transform(
                data_dict, 
                np.array([dx, 0, 0, 0, 0, 0, 0, dx, 0, 0])
            ))
            from_left.append(data_dict["observations"]["state"][0] - self.workspace_width)
            from_center.append(data_dict["observations"]["state"][0] - rx)
        
        print(self.workspace_width)
        print(from_left)
        print(from_center)
        