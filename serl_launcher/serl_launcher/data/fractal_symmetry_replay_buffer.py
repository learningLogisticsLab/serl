import gym
import numpy as np
from serl_launcher.data.dataset import DatasetDict
from serl_launcher.data.replay_buffer import ReplayBuffer
import copy

from datetime import datetime as dt
class FractalSymmetryReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        branch_method: str,
        split_method: str,
        workspace_width: int,
        kwargs: dict
    ):
        self.current_branch_count=1
        self.update_max_traj_length = False

        method_check = "split_method"
        match split_method:
            case "time":
                assert "max_depth" in kwargs.keys(), self._handle_bad_args_(method_check, split_method, "max_depth")
                self.max_depth=kwargs["max_depth"]
                del kwargs["max_depth"]

                assert "max_traj_length" in kwargs.keys(), self._handle_bad_args_(method_check, split_method, "max_traj_length")
                self.max_traj_length=kwargs["max_traj_length"]

                assert "alpha" in kwargs.keys(), self._handle_bad_args_(method_check, split_method, "alpha")
                self.alpha=kwargs["alpha"]
                update_max_traj_length = True
                self.split = self.time_split 

            case "constant":
                self.split = self.constant_split

            case "test":
                self.split = self.test_split
            case _:
                raise ValueError("incorrect value passed to split_method")
        
        method_check = "branch_method"
        match branch_method:
            case "fractal":

                assert "branching_factor" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "branching_factor")
                self.branching_factor=kwargs["branching_factor"]
                del kwargs["branching_factor"]

                self.branch = self.fractal_branch

            # case "linear":
            #     assert "branch_count_rate_of_change" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "branch_count_rate_of_change")
            #     self.branch_count_rate_of_change=kwargs["branch_count_rate_of_change"]
            #     del kwargs["branch_count_rate_of_change"]

            #     assert "starting_branch_count" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "starting_branch_count")
            #     self.current_branch_count = kwargs["starting_branch_count"]
            #     del kwargs["starting_branch_count"]

            #     self.branch = self.linear_branch

            case "constant":
                assert "starting_branch_count" in kwargs.keys(), self._handle_bad_args_("branch_method", branch_method, "starting_branch_count")
                self.current_branch_count = kwargs["starting_branch_count"]
                del kwargs["starting_branch_count"]

                self.branch = self.constant_branch

            case "test":
                self.branch = self.test_branch

            case _:
                raise ValueError("incorrect value passed to branch_method")
        
        self.x_obs_idx = kwargs["x_obs_idx"]
        self.y_obs_idx = kwargs["y_obs_idx"]
        self.workspace_width = workspace_width
        self.timestep = 0
        self.current_depth = 0
        self.start_num = kwargs["start_num"]

        self.branch_index = np.empty(self.current_branch_count, dtype=np.float32)
        constant = self.workspace_width/(2 * self.current_branch_count)
        for i in range(0, self.current_branch_count):
            self.branch_index[i] = (2 * i + 1) * constant

        for k in kwargs.keys():
            print(f"\033[33mWARNING \033[0m argument \"{k}\" not used")
        
        # TODO
        #   Create more methods with tests
        # 
        # Future considerations:
        #   when dealing with x and y for transforms, consider two versions:
        #       radial trees at different angles can have strong fractal symmetry built in but will be more difficult to evenly space lowest level transforms
        #       grid-based will inherently evenly space lowest transforms, but fractal symmetry will be restricted to 9 branching_factor (original plus 8) in order to conform

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
        )

    def _handle_bad_args_(self, type: str, method: str, arg: str) :
        return f"\033[31mERROR: \033[0m{arg} must be defined for {type} \"{method}\""
    

    def transform(self, data_dict: DatasetDict, transform: np.array):
        #   return data_dict with positional arguments += translation
        data_dict["observations"] += transform
        data_dict["next_observations"] += transform
        
    # FOR TESTING ONLY
    def test_branch(self):
        self.current_depth += 1
        return 1
    
    def fractal_branch(self):
        # return a new number of branches = branching_factor ^ depth
        return self.branching_factor ** self.current_depth
    
    def constant_branch(self):
        # return current number of branches
        return self.current_branch_count
    
    # REQUIRES TESTING
    # def linear_branch(self):
    #     # return a new number of branches = branches_count + n
    #     return self.current_branch_count + self.branch_count_rate_of_change
    
    # FOR TESTING ONLY
    def test_split(self, data_dict: DatasetDict):
        return True
            
    # REQUIRES TESTING
    def time_split(self, data_dict: DatasetDict):
        if self.timestep % (self.max_traj_length//self.max_depth) or self.current_depth >= self.max_depth:
            return False
        self.current_depth += 1
        return True
    
    def constant_split(self, data_dict: DatasetDict):
        return True
    
    def insert(self, data_dict_not: DatasetDict):

        sector_1 = dt.now()
        data_dict = copy.deepcopy(data_dict_not)

        # Update number of branches if needed
        if self.split(data_dict):
            temp = self.current_branch_count
            self.current_branch_count = self.branch()
            if temp != self.current_branch_count:
                self.branch_index = np.empty(self.current_branch_count, dtype=np.float32)
                constant = self.workspace_width/(2 * self.current_branch_count)
                for i in range(0, self.current_branch_count):
                    self.branch_index[i] = (2 * i + 1) * constant
        
        sector_2 = dt.now()
        # Initialize to extreme x and y
        x = -self.workspace_width/2
        transform = np.zeros_like(data_dict["observations"])
        transform[self.x_obs_idx] = x
        transform[self.y_obs_idx] = x
        self.transform(data_dict, transform)
        sector_3 = dt.now()
        # Transform and insert transitions (multiprocessing in the future)
        for x in range(0, self.current_branch_count):
            transform[self.x_obs_idx] = self.branch_index[x]
            for y in range(0, self.current_branch_count):
                new_data_dict = copy.deepcopy(data_dict)
                transform[self.y_obs_idx] = self.branch_index[y]
                self.transform(new_data_dict, transform)
                super().insert(new_data_dict)
        sector_4 = dt.now()
        self.timestep += 1
        if data_dict["dones"]:
            self.current_depth = 0
            if self.update_max_traj_length:
                self.max_traj_length = int(self.timestep * self.alpha + self.max_traj_length * (1 - self.alpha))
            self.timestep = 0

        finish = dt.now()
        #print(f"Splits: {(sector_2 - sector_1).total_seconds():.5f} : {(sector_3 - sector_2).total_seconds():.5f} : {(sector_4 - sector_3).total_seconds():.5f} : {(finish - sector_4).total_seconds():.5f}\nLaptime: {(finish - sector_1).total_seconds():.5f}")
