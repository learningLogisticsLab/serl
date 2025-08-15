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
        workspace_width: int,
        x_obs_idx : np.ndarray,
        y_obs_idx : np.ndarray,
        branch_method: str,
        split_method: str,
        img_keys: list,
        kwargs: dict,
    ):
        
        # Initialize values
        self.debug_time = True
        self.current_branch_count = 1
        self.update_max_traj_length = False
        self.workspace_width = workspace_width
        self.img_keys = img_keys
        self._img_insert_index_ = 0
            
        # Set the idx value (changes depending on environment/wrapper) of the x and y observations and next_observations
        self.x_obs_idx = x_obs_idx
        self.y_obs_idx = y_obs_idx

        # Set initial fractal config values
        self.timestep = 0
        self.current_depth = 0

        self.split_method = split_method
        self.branch_method = branch_method

        self._handle_methods_(kwargs)

        # Warn about unused kwargs
        for k in kwargs.keys():
            print(f"\033[33mWARNING \033[0m argument \"{k}\" not used")
        
        # Account for images
        if self.img_keys:
            img_buffer_size = (capacity + self.expected_branches - 1) // self.expected_branches + 1
            self.img_buffer = {}
            self.insert = self.insert_with_images
        for k in img_keys:
            temp = np.empty(shape=observation_space.spaces[k].shape, dtype=observation_space.spaces[k].dtype)
            temp = temp[np.newaxis, :]
            self.img_buffer[k] = np.repeat(temp, img_buffer_size, axis=0)
            
            observation_space.spaces[k] = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(1,), dtype=np.int32)

        # Init replay buffer class
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
        )

        for k in img_keys:
            self.dataset_dict["observations"][k] = self.dataset_dict["observations"][k].flatten()
            self.dataset_dict["next_observations"][k] = self.dataset_dict["next_observations"][k].flatten()

        self.generate_transform_deltas()

    def _handle_method_arg_(self, value, method_type, method, kwargs):
        if hasattr(self, value):
            return
        assert value in kwargs.keys(), f"\033[31mERROR: \033[0m{value} must be defined for {method_type} \"{method}\""
        setattr(FractalSymmetryReplayBuffer, value, kwargs[value])
        del kwargs[value]

    def _handle_methods_(self, kwargs):
        
        # Initialize branch_method
        match self.branch_method:
            case "fractal":
                self._handle_method_arg_("max_depth", "branch_method", self.branch_method, kwargs)
                self._handle_method_arg_("branching_factor", "branch_method", self.branch_method, kwargs)

                self.branch = self.fractal_branch
                if not self.split_method:
                    self.split_method = "time"
                self.expected_branches = self.branching_factor ** self.max_depth
            
            case "contraction":
                self._handle_method_arg_("max_depth", "branch_method", self.branch_method, kwargs)
                self._handle_method_arg_("branching_factor", "branch_method", self.branch_method, kwargs)

                self.branch = self.fractal_contraction
                if not self.split_method:
                    self.split_method = "time"
                self.expected_branches = self.branching_factor ** self.max_depth
            
            case "linear":
                raise NotImplementedError("linear branch method is not yet implemented")
                # self.branch = self.linear_branch
                

            case "disassociated":
                self._handle_method_arg_("min_branch_count", "branch_method", self.branch_method, kwargs)
                self._handle_method_arg_("max_branch_count", "branch_method", self.branch_method, kwargs)

                if self.min_branch_count > self.max_branch_count:
                    raise ValueError(f"min_branch_count ({self.min_branch_count}) is larger than max_branch_count ({self.max_branch_count})")

                match kwargs["disassociated_type"]:
                    case "hourglass":
                        self.starting_branch_count = self.max_branch_count
                    case "octahedron":
                        self.starting_branch_count = self.min_branch_count
                    case _:
                        raise ValueError(f"incorrect value passed to disassociated_type")
                
                self.disassociated_type = kwargs["disassociated_type"]
                del kwargs["disassociated_type"]
                self.branch = self.disassociated_branch
                if not self.split_method:
                    self.split_method = "time"
                self.expected_branches = self.max_branch_count
            
            case "constant":
                self._handle_method_arg_("starting_branch_count", "branch_method", self.branch_method, kwargs)

                self.branch = self.constant_branch
                if not self.split_method:
                    self.split_method = "never"
                self.expected_branches = self.starting_branch_count
            
            case _:
                raise ValueError("incorrect value passed to branch_method")

        match self.split_method:
            case "time":
                self._handle_method_arg_("max_depth", "split_method", self.split_method, kwargs)
                self._handle_method_arg_("max_traj_length", "split_method", self.split_method, kwargs)
                self._handle_method_arg_("alpha", "split_method", self.split_method, kwargs)
                
                self.update_max_traj_length = True
                self.split = self.time_split 

            case "constant":
                self.split = self.constant_split
            
            case "never":
                self.split = self.never_split
                
            case _:
                raise ValueError("incorrect value passed to split_method")
        
        if hasattr(self, "starting_branch_count"):
            self.current_branch_count = self.starting_branch_count
    
    def generate_transform_deltas(self):
        
        obs_state = self.dataset_dict["observations"]
        if self.img_keys:
            obs_state = self.dataset_dict["observations"]["state"]

        obs_size = len(obs_state[0])
        total_branches = self.current_branch_count ** 2

        self.transform_deltas = np.zeros(shape=(total_branches, obs_size), dtype=np.float32)

        idx = np.arange(total_branches)
        x_deltas, y_deltas = np.divmod(idx, self.current_branch_count)

        x_deltas = (2 * x_deltas + 1) * self.workspace_width / (2 * self.current_branch_count)
        y_deltas = (2 * y_deltas + 1) * self.workspace_width / (2 * self.current_branch_count)
        x_deltas = np.repeat(x_deltas, self.x_obs_idx.size)
        y_deltas = np.repeat(y_deltas, self.y_obs_idx.size)
        x_deltas = np.reshape(x_deltas, (total_branches, self.x_obs_idx.size))
        y_deltas = np.reshape(y_deltas, (total_branches, self.y_obs_idx.size))

        self.transform_deltas[:, self.x_obs_idx] = x_deltas
        self.transform_deltas[:, self.y_obs_idx] = y_deltas
    
    def fractal_branch(self):
        '''
        Computes the number of branches for the current depth using an exponential growth rule.

        This method implements a "fractal branching" strategy, where the number of branches
        increases exponentially with depth. At each depth `d`, the number of branches is calculated as:

            num_branches = branching_factor ** current_depth

        where:
            - branching_factor: The base number of branches at each split.
            - current_depth: The current depth in the fractal tree (self.current_depth).

        Returns:
            int: The computed number of branches for the current depth.
        '''        
        # return a new number of branches = branching_factor ^ depth
        return self.branching_factor ** self.current_depth
    
    def fractal_contraction(self):
        '''
        Computes the number of branches for the current depth using a contraction rule.

        This method implements a "fractal contraction" branching strategy, where the number
        of branches decreases exponentially with depth. At each depth `d`, the number of branches
        is calculated as:

            num_branches = start_num / (branching_factor ** (d - 1))

        where:
            - start_num: The initial number of branches at depth 1.
            - branching_factor: The factor by which the number of branches contracts at each depth.
            - d: The current depth (self.current_depth).

        Returns:
            int: The computed number of branches for the current depth.
        '''

        return self.branching_factor ** (self.max_depth - self.current_depth + 1)
    
    def constant_branch(self):
        '''
        Used to create pure translations with no further branching.
        self.current_branch_count used to set the total number of transformations.
        '''
        # return current number of branches
        return self.current_branch_count
    
    def disassociated_branch(self):
        '''
        Used to create branches for disassociated fractal methods.
        self.min_branch_count specifies the mininum branch count desired during the fractal rollout
        self.max_branch_count specifies the maximum branch count desired during the fractal rollout
        self.disassociated_type specifies whether to expand and then contract or to contract and then expand
        self.steps_per_depth specifies the number of timesteps to take before splitting 
                (calculated indirectly via self.max_traj_length / self.num_depth_sectors)
        self.num_depth_sectors specifies the number of sectors the rollout should be divided into for even splitting
        '''
        if self.disassociated_type == "hourglass":
            return int((self.max_branch_count - self.min_branch_count)/(self.max_depth/2) * np.abs(self.current_depth - (self.max_depth/2)) + self.min_branch_count)
        elif self.disassociated_type == "octahedron":
            return int((self.min_branch_count - self.max_branch_count)/(self.max_depth/2) * np.abs(self.current_depth - (self.max_depth/2)) + self.max_branch_count)
        
    def linear_branch(self):
        # return a new number of branches = branches_count + n
        return self.current_branch_count + self.branching_factor
            
    def time_split(self, data_dict: DatasetDict):
        if self.timestep % (self.max_traj_length//self.max_depth) or self.current_depth >= self.max_depth:
            return False
        self.current_depth += 1
        return True 

    def constant_split(self, data_dict: DatasetDict):
        self.current_depth += 1
        return True
    
    def never_split(self, data_dict: DatasetDict):
        return False

    def insert_images(self, observation: dict):
        for k in self.img_keys:
            self.img_buffer[k][self._img_insert_index_] = observation[k]
        self._img_insert_index_ += 1

    def insert(self, data: DatasetDict):

        data_dict = copy.deepcopy(data)

        # Update number of branches if needed
        if self.split(data_dict):
            temp = self.current_branch_count
            self.current_branch_count = self.branch()
            # Update transform_deltas if needed
            if temp != self.current_branch_count:
                self.generate_transform_deltas()

        # Initialize to extreme x and y
        base_diff = -self.workspace_width/2
        data_dict["observations"][self.x_obs_idx] += base_diff
        data_dict["observations"][self.y_obs_idx] += base_diff
        data_dict["next_observations"][self.x_obs_idx] += base_diff
        data_dict["next_observations"][self.y_obs_idx] += base_diff

        # Transform and insert transitions
        num_transforms = self.current_branch_count ** 2
        obs_batch = np.tile(data_dict["observations"], (num_transforms, 1))
        next_obs_batch = np.tile(data_dict["next_observations"], (num_transforms, 1))

        obs_batch += self.transform_deltas
        next_obs_batch += self.transform_deltas

        
        data_dict["observations"] = obs_batch
        data_dict["next_observations"] = next_obs_batch
        data_dict["actions"] = np.tile(data_dict["actions"], (num_transforms, 1))
        data_dict["rewards"] = np.tile(data_dict["rewards"], num_transforms)
        data_dict["masks"] = np.tile(data_dict["masks"], num_transforms)
        data_dict["dones"] = np.tile(data_dict["dones"], num_transforms)

        super().insert(data_dict, batch_size=num_transforms)

        # Reset current_depth, timestep, and max_traj_length
        self.timestep += 1
        if data_dict["dones"][0]:
            self.current_depth = 0
            if self.update_max_traj_length:
                self.max_traj_length = int(self.timestep * self.alpha + self.max_traj_length * (1 - self.alpha))
            self.timestep = 0

    def insert_with_images(self, data: DatasetDict):

        data_dict = copy.deepcopy(data)
        
        # Update number of branches if needed
        if self.split(data_dict):
            temp = self.current_branch_count
            self.current_branch_count = self.branch()
            # Update transform_deltas if needed
            if temp != self.current_branch_count:
                self.generate_transform_deltas()
        
        # Insert images
        if self.timestep == 0:
            self.insert_images(data_dict["observations"])
        self.insert_images(data_dict["next_observations"])

        for k in self.img_keys:
            data_dict["observations"][k] = (self._img_insert_index_ - 2) % len(self.img_buffer[k])
            data_dict["next_observations"][k] = (self._img_insert_index_ - 1) % len(self.img_buffer[k])
                
        # Initialize to extreme x and y
        base_diff = -self.workspace_width/2
        data_dict["observations"]["state"][self.x_obs_idx] += base_diff
        data_dict["observations"]["state"][self.y_obs_idx] += base_diff
        data_dict["next_observations"]["state"][self.x_obs_idx] += base_diff
        data_dict["next_observations"]["state"][self.y_obs_idx] += base_diff
        
        # Transform and insert transitions
        num_transforms = self.current_branch_count ** 2
        obs_batch = np.tile(data_dict["observations"]["state"], (num_transforms, 1))
        next_obs_batch = np.tile(data_dict["next_observations"]["state"], (num_transforms, 1))

        obs_batch += self.transform_deltas
        next_obs_batch += self.transform_deltas

        
        data_dict["observations"]["state"] = obs_batch
        data_dict["next_observations"]["state"] = next_obs_batch
        data_dict["actions"] = np.tile(data_dict["actions"], (num_transforms, 1))
        data_dict["rewards"] = np.tile(data_dict["rewards"], num_transforms)
        data_dict["masks"] = np.tile(data_dict["masks"], num_transforms)
        data_dict["dones"] = np.tile(data_dict["dones"], num_transforms)

        for k in self.img_keys:
            data_dict["observations"][k] = np.tile(data_dict["observations"][k], num_transforms)
            data_dict["next_observations"][k] = np.tile(data_dict["next_observations"][k], num_transforms)

        super().insert(data_dict, batch_size=num_transforms)

        # Reset current_depth, timestep, and max_traj_length
        self.timestep += 1
        if data_dict["dones"][0]:
            self.current_depth = 0
            if self.update_max_traj_length:
                self.max_traj_length = int(self.timestep * self.alpha + self.max_traj_length * (1 - self.alpha))
            self.timestep = 0