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
        branch_method: str,
        split_method: str,
        workspace_width: int,
        **kwargs: dict
    ):
        self.current_branch_count=1

        match split_method:
            case "time":
                assert "timesplit_freq" in kwargs.keys(), "\033[31mERROR \033[0mtimesplit_freq must be defined for split_method \"time\""
                self.timesplit_freq=kwargs["timesplit_freq"]
                del kwargs["timesplit_freq"]
                self.split = self.time_split
            case "rel_pos":
                print("NOT IMPLEMENTED")
                self.split = self.rel_pos_split
            case "height":
                print("NOT IMPLEMENTED")
                self.split = self.height_split
            case "velocity":
                print("NOT IMPLEMENTED")
                self.split = self.velocity_split
            case "test":
                self.split = self.test_split
            case _:
                raise ValueError("incorrect value passed to split_method")
            
        match branch_method:
            case "fractal":
                assert "depth" in kwargs.keys(), "\033[31mERROR \033[0mdepth must be defined for branch_method \"fractal\""
                self.depth=kwargs["depth"]
                del kwargs["depth"]

                assert "dendrites" in kwargs.keys(), "\033[31mERROR \033[0mdendrites must be defined for branch_method \"fractal\""
                self.dendrites=kwargs["dendrites"]
                del kwargs["dendrites"]

                self.branch = self.fractal_branch

            case "linear":
                assert "branch_count_rate_of_change" in kwargs.keys(), "\033[31mERROR \033[0mbranch_count_rate_of_change must be defined for branch_method \"fractal\""
                self.branch_count_rate_of_change=kwargs["branch_count_rate_of_change"]
                del kwargs["branch_count_rate_of_change"]

                assert "starting_branch_count" in kwargs.keys(), "\033[31mERROR \033[0mstarting_branch_count must be defined for branch_method \"constant\""
                self.current_branch_count = kwargs["starting_branch_count"]
                del kwargs["starting_branch_count"]

                self.branch = self.linear_branch

            case "constant":
                assert "starting_branch_count" in kwargs.keys(), "\033[31mERROR \033[0mstarting_branch_count must be defined for branch_method \"constant\""
                self.current_branch_count = kwargs["starting_branch_count"]
                del kwargs["starting_branch_count"]

                self.branch = self.constant_branch

            case "test":
                self.branch = self.test_branch

            case _:
                raise ValueError("incorrect value passed to branch_method")

        for i in kwargs.keys():
            print(f"\033[33mWARNING \033[0m {i} argument not used")
        
        
        
        self.workspace_width = workspace_width
        self.current_depth = 0
        
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

    def transform(self, data_dict: DatasetDict, translation: np.array):
        #   return data_dict with positional arguments += translation
        data_dict["observations"] += translation
        data_dict["next_observations"] += translation
        
    # FOR TESTING ONLY
    def test_branch(self):
        # self.current_branch_count = np.random.randint(2500)
        # while(not self.current_branch_count % 2):
        #     self.current_branch_count = np.random.randint(2500)
        # return self.current_branch_count
        return 81
    
    def fractal_branch(self):
        # return a new number of branches = dendrites ^ depth
        self.current_depth += 1
        return self.dendrites ** (self.current_depth)
    
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
            
    # NOT IMPLEMENTED
    def time_split(self, data_dict: DatasetDict):
        # return True when a set time has passed
        raise NotImplementedError
    
    # NOT IMPLEMENTED
    def rel_pos_split(self, data_dict: DatasetDict):
        # return True if there is a change of depth determined by relative position
        raise NotImplementedError
    
    # NOT IMPLEMENTED
    def height_split(self, data_dict: DatasetDict):
        # return True if there is a change of depth determined by height
        raise NotImplementedError
    
    # NOT IMPLEMENTED
    def velocity_split(self, data_dict: DatasetDict):
        # return True if there is a change of depth determined by velocity
        raise NotImplementedError
    
    def constant_split(self, data_dict: DatasetDict):
        # return True every transition
        return True
    
    def insert(self, data_dict: DatasetDict):
        
        # Update number of branches if needed
        if self.split(data_dict):
            self.current_branch_count = self.branch()
        
        # Transform and insert branches
        dx = self.workspace_width/self.current_branch_count
        x = (-self.workspace_width + dx)/2
        self.transform(data_dict, np.array([x, 0, 0, 0, 0, 0, 0, x, 0, 0]))

        for t in range(0, self.current_branch_count):
            super().insert(data_dict)
            self.transform(data_dict, np.array([dx, 0, 0, 0, 0, 0, 0, dx, 0, 0]))
            
            
