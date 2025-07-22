import gym
import numpy as np
from serl_launcher.data.dataset import DatasetDict
from serl_launcher.data.replay_buffer import ReplayBuffer
import copy

import time

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

        method_check = "split_method"
        match split_method:
            case "time":
                assert "depth" in kwargs.keys(), self._handle_bad_args_(method_check, split_method, "depth")
                self.max_depth=kwargs["depth"]
                del kwargs["depth"]

                assert "max_steps" in kwargs.keys(), self._handle_bad_args_(method_check, split_method, "max_steps")
                self.split = self.time_split 

            case "test":
                self.split = self.test_split
            case _:
                raise ValueError("incorrect value passed to split_method")
        
        method_check = "branch_method"
        match branch_method:
            case "fractal":
                assert "depth" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "depth")
                self.max_depth=kwargs["depth"]
                del kwargs["depth"]

                assert "dendrites" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "dendrites")
                self.dendrites=kwargs["dendrites"]
                del kwargs["dendrites"]

                self.branch = self.fractal_branch

            case "linear":
                assert "branch_count_rate_of_change" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "branch_count_rate_of_change")
                self.branch_count_rate_of_change=kwargs["branch_count_rate_of_change"]
                del kwargs["branch_count_rate_of_change"]

                assert "starting_branch_count" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "starting_branch_count")
                self.current_branch_count = kwargs["starting_branch_count"]
                del kwargs["starting_branch_count"]

                self.branch = self.linear_branch

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
        self.current_depth = 1
        self.counter = 1
        self.branch_index = [workspace_width/2]

        for k in kwargs.keys():
            print(f"\033[33mWARNING \033[0m argument \"{k}\" not used")
        
        # TODO
        #   Add flags for variables passed to
        #   Create more methods with tests
        # 
        # Future considerations:
        #   consider different strategy for better accuracy
        #   when dealing with x and y for transforms, consider two versions:
        #       radial trees at different angles can have strong fractal symmetry built in but will be more difficult to evenly space lowest level transforms
        #       grid-based will inherently evenly space lowest transforms, but fractal symmetry will be restricted to 9 dendrites (original plus 8) in order to conform

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
        )

    def _handle_bad_args_(self, type: str, method: str, arg: str) :
        return f"\033[31mERROR: \033[0m{arg} must be defined for {type} \"{method}\""
    

    def transform(self, data_dict: DatasetDict, transform: np.array):
        #   return data_dict with positional arguments += translation
        assert data_dict["observations"].shape == transform.shape, "transform broke at observations"
        assert data_dict["observations"].shape == transform.shape, "transform broke at next_observations"
        data_dict["observations"] += transform
        data_dict["next_observations"] += transform
        
    # FOR TESTING ONLY
    def test_branch(self):
        # self.current_branch_count = np.random.randint(2500)
        # while(not self.current_branch_count % 2):
        #     self.current_branch_count = np.random.randint(2500)
        # return self.current_branch_count
        return 1
    
    def fractal_branch(self):
        # return a new number of branches = dendrites ^ depth
        temp = self.dendrites ** self.current_depth
        self.current_depth += 1
        if self.current_depth > self.max_depth:
            return self.dendrites ** self.max_depth
        return temp
    
    # REQUIRES TESTING
    def constant_branch(self):
        # return current number of branches
        return self.current_branch_count
    
    # REQUIRES TESTING
    def linear_branch(self):
        # return a new number of branches = branches_count + n
        return self.current_branch_count + self.branch_count_rate_of_change
    
    # FOR TESTING ONLY
    def test_split(self, data_dict: DatasetDict):
        return True
            
    # NOT IMPLEMENTED (HARDCODED)
    def time_split(self, data_dict: DatasetDict):
        # return True when a set time has passed
        match self.counter % 12:
            case 0:
                return True
            case _:
                return False
    
    def constant_split(self, data_dict: DatasetDict):
        # return True every transition
        return True
    
    def insert(self, data_dict_not: DatasetDict):
        
        data_dict = copy.deepcopy(data_dict_not)
        # start_time = time.time()
        # Update number of branches if needed
        if self.split(data_dict):
            temp = self.current_branch_count
            self.current_branch_count = self.branch()
            if temp != self.current_branch_count:
                self.branch_index = np.empty(self.current_branch_count, dtype=np.float32)
                constant = self.workspace_width/(2 * self.current_branch_count)
                for i in range(0, self.current_branch_count):
                    self.branch_index[i] = (2 * i + 1) * constant
        # rb_origin = [data_dict["observations"][self.x_obs_idx[0]], data_dict["observations"][self.y_obs_idx[0]]]
        # block_origin = [data_dict["observations"][self.x_obs_idx[1]], data_dict["observations"][self.y_obs_idx[1]]]
        # rb_next_origin = [data_dict["next_observations"][self.x_obs_idx[0]], data_dict["next_observations"][self.y_obs_idx[0]]]
        # block_next_origin = [data_dict["next_observations"][self.x_obs_idx[1]], data_dict["next_observations"][self.y_obs_idx[1]]]

        # Initialize to extreme x and y
        x = -self.workspace_width/2
        transform = np.zeros_like(data_dict["observations"])
        transform[self.x_obs_idx] = x
        transform[self.y_obs_idx] = x
        self.transform(data_dict, transform)

        # rb_pos_print = np.empty(shape=(self.current_branch_count, self.current_branch_count, 2), dtype=np.float32)
        # block_pos_print = np.empty(shape=(self.current_branch_count, self.current_branch_count, 2), dtype=np.float32)
        # rb_next_pos_print = np.empty(shape=(self.current_branch_count, self.current_branch_count, 2), dtype=np.float32)
        # block_next_pos_print = np.empty(shape=(self.current_branch_count, self.current_branch_count, 2), dtype=np.float32)
        # Transform and insert transitions (multiprocessing in the future)
        for x in range(0, self.current_branch_count):
            x_diff = self.branch_index[x]
            transform[self.x_obs_idx] = x_diff
            for y in range(0, self.current_branch_count):
                new_data_dict = copy.deepcopy(data_dict)
                y_diff = self.branch_index[y]
                transform[self.y_obs_idx] = y_diff
                self.transform(new_data_dict, transform)

                # keep info for debug
                # rb_pos_print[y][x][0] = round(new_data_dict["observations"][self.x_obs_idx[0]], 2)
                # rb_pos_print[y][x][1] = round(new_data_dict["observations"][self.y_obs_idx[0]], 2)
                # block_pos_print[y][x][0] = round(new_data_dict["observations"][self.x_obs_idx[1]], 2)
                # block_pos_print[y][x][1] = round(new_data_dict["observations"][self.y_obs_idx[1]], 2)
                # rb_next_pos_print[y][x][0] = round(new_data_dict["next_observations"][self.x_obs_idx[0]], 2)
                # rb_next_pos_print[y][x][1] = round(new_data_dict["next_observations"][self.y_obs_idx[0]], 2)
                # block_next_pos_print[y][x][0] = round(new_data_dict["next_observations"][self.x_obs_idx[1]], 2)
                # block_next_pos_print[y][x][1] = round(new_data_dict["next_observations"][self.y_obs_idx[1]], 2)

                # absolutely make sure nothing wrong is being changed
                # assert data_dict["rewards"] == new_data_dict["rewards"]
                # assert (data_dict["actions"] == new_data_dict["actions"]).all()
                # assert data_dict["dones"] == new_data_dict["dones"]
                # assert data_dict["masks"] == new_data_dict["masks"]
                # for i in range(0, data_dict["observations"].size):
                    # if i in self.x_obs_idx or i in self.y_obs_idx:
                        # continue
                # assert data_dict["observations"][i] == new_data_dict["observations"][i]
                # assert data_dict["next_observations"][i] == new_data_dict["next_observations"][i]

                super().insert(new_data_dict)

        self.counter += 1

        # print(f"original x,y of hand before transition: {rb_origin}")
        # print(f"transforms: \n{rb_pos_print}")
        # print(f"original x,y of hand after transition: {rb_next_origin}")
        # print(f"transforms: \n{rb_next_pos_print}")
        # print(f"original x,y of block before transition: {block_origin}")
        # print(f"transforms: \n{block_pos_print}")
        # print(f"original x,y of block after transition: {block_next_origin}")
        # print(f"transforms: \n{block_next_pos_print}")

        if data_dict["dones"]:
            self.counter = 1
            self.current_depth = 1
        # end_time = time.time()
        # duration = end_time - start_time
        # print(f"#{self._insert_index - 1} Duration: {duration:.4f}")

            