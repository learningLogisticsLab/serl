import gym
import numpy as np
from serl_launcher.data.dataset import DatasetDict
from serl_launcher.data.replay_buffer import ReplayBuffer

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
        depth: int,
        dendrites: int, 
        timesplit_freq: int,
        branch_count_rate_of_change: int,
    ):

        self.split_method=split_method # Determines method for when to change transformation number
        self.branch_method=branch_method # Determines method for how many transforms to make for a given transition
        self.workspace_width = workspace_width # Total 1D distance for transform calculations

        # Potentially temporary variables
        self.transition_num = 0
        self.current_branch_count = 1 # Current number of branches
        self.current_depth = 0 # Current depth
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
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
        )

    # REQUIRES TESTING
    def transform(self, data_dict: DatasetDict, translation: np.array):
        #   return data_dict with positional arguments += translation
        data_dict["observations"] += translation
        data_dict["next_observations"] += translation
    
    # REQUIRES TESTING
    def fractal_branch(self, data_dict: DatasetDict):
        # return a new number of branches = dendrites ^ depth
        self.current_depth += 1
        return self.dendrites ** (self.current_depth)
    
    # REQUIRES TESTING
    def constant_branch(self, data_dict: DatasetDict):
        # return current number of branches
        return self.current_branch_count
    
    # REQUIRES TESTING
    def linear_branch(self, data_dict: DatasetDict):
        # return a new number of branches = branches_count + n
        return self.current_branch_count + self.branch_count_rate_of_change
    
    # FOR TESTING ONLY
    def test_split(self, data_dict: DatasetDict):
        split = False
        if self.transition_num % 2:
            split = True
        return split
            
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
            case "test":
                split = self.test_split
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
        
        self.transition_num += 1
        # Update number of branches if needed
        if split(data_dict):
            self.current_branch_count = branch(data_dict)
        
        #---------------------TEMPORARY SECTION TO BE IMPROVED------------------------#
        # Save positions
        rx = data_dict["observations"][0]
        # ry = data_dict["observations"]["state"][1]
        bx = data_dict["observations"][7]
        # by = data_dict["observations"]["state"][8]

        rx2 = data_dict["next_observations"][0]
        # ry2 = data_dict["next_observations"]["state"][1]
        bx2 = data_dict["next_observations"][7]
        # by2 = data_dict["next_observations"]["state"][8]

        # OR set to extreme for iterations
        x = self.workspace_width / 2
        data_dict["observations"] -= np.array([x, 0, 0, 0, 0, 0, 0, x, 0, 0])
        data_dict["next_observations"] -= np.array([x, 0, 0, 0, 0, 0, 0, x, 0, 0])
        #-----------------------------------------------------------------------------#

        from_left = []
        from_center = []

        # Initial insert of most-extreme branch
        dx = self.workspace_width / (self.current_branch_count * 2)
        self.transform(data_dict, np.array([dx, 0, 0, 0, 0, 0, 0, dx, 0, 0]))
        super().insert(data_dict)
        dx = dx * 2

        from_left.append(data_dict["observations"][0] - self.workspace_width / 2)
        from_center.append(data_dict["observations"][0] - rx)

        # Insert of rest of branches
        for t in range(0, self.current_branch_count - 1):
            self.transform(data_dict, np.array([dx, 0, 0, 0, 0, 0, 0, dx, 0, 0]))
            super().insert(data_dict)
            from_left.append(data_dict["observations"][0] - self.workspace_width / 2)
            from_center.append(data_dict["observations"][0] - rx)
        
        print("\ntransition:\t", self.transition_num,"\ndepth:\t\t", self.current_depth, sep="\t")
        print("workspace width:", self.workspace_width, sep="\t")
        print("transforms from '0':", np.round(from_left, 2), sep="\t")
        print("transforms from center:", np.round(from_center, 2), "\n", sep="\t")
        return
