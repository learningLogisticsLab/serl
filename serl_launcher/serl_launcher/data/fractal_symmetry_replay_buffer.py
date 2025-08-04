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
        workspace_width_method: str,
        kwargs: dict
    ):
        
        # Initialize values
        self.debug_time = False
        self.current_branch_count=1
        self.update_max_traj_length = False
        self.workspace_width = workspace_width

        # Set correct split function in self.split pointer.
        method_check = "split_method"
        match split_method:
            case "time":
                assert "max_depth" in kwargs.keys(), self._handle_bad_args_(method_check, split_method, "max_depth")
                self.max_depth=kwargs["max_depth"]

                assert "max_traj_length" in kwargs.keys(), self._handle_bad_args_(method_check, split_method, "max_traj_length")
                self.max_traj_length=kwargs["max_traj_length"]

                assert "alpha" in kwargs.keys(), self._handle_bad_args_(method_check, split_method, "alpha")
                self.alpha=kwargs["alpha"]
                update_max_traj_length = True
                self.split = self.time_split 

            case "constant":
                self.split = self.constant_split

            case "disassociated":
                assert "max_traj_length" in kwargs.keys(), self._handle_bad_args_(method_check, split_method, "max_traj_length")
                self.max_traj_length=kwargs["max_traj_length"]
                self.update_max_traj_length = True

                assert "alpha" in kwargs.keys(), self._handle_bad_args_(method_check, split_method, "alpha")
                self.alpha=kwargs["alpha"]

                assert "num_depth_sectors" in kwargs.keys(), self._handle_bad_args_(method_check, split_method, "num_depth_sectors")
                self.num_depth_sectors=kwargs["num_depth_sectors"]

                self.steps_per_depth = int(np.floor(self.max_traj_length/self.num_depth_sectors)) # does this need to be np.float32?

                self.transition_counter = 0

                self.split = self.disassociated_split
                

            case "test":
                self.split = self.test_split
            case _:
                raise ValueError("incorrect value passed to split_method")
        
        # Set correct branch function in self.branch pointer.
        method_check = "branch_method"
        match branch_method:
            case "fractal":
                assert "max_depth" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "max_depth")
                self.max_depth=kwargs["max_depth"]

                assert "branching_factor" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "branching_factor")
                self.branching_factor=kwargs["branching_factor"]

                assert "start_num" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "start_num")
                self.start_num=kwargs["start_num"]

                self.branch = self.fractal_branch
            
            case "contraction":
                assert "start_num" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "start_num")
                self.start_num=kwargs["start_num"]

                # Add branching_factor for contraction method
                assert "branching_factor" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "branching_factor")
                self.branching_factor = kwargs["branching_factor"]

                self.branch = self.fractal_contraction
            
            case "linear":
                raise NotImplementedError("linear branch method is not yet implemented")
                # self.branch = self.linear_branch

            case "disassociated":
                assert "min_branch_count" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "min_branch_count)")
                self.min_branch_count = kwargs["min_branch_count"]

                assert "max_branch_count" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "max_branch_count)")
                self.max_branch_count = kwargs["max_branch_count"]

                if self.min_branch_count > self.max_branch_count:
                    raise ValueError(f"min_branch_count: {self.min_branch_count} is larger than max_branch_count: {self.max_branch_count}. Max should be larger than min.")

                assert "disassociated_type" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "disassociated_type")
                if kwargs["disassociated_type"] == "hourglass":
                    self.current_branch_count = self.max_branch_count
                elif kwargs["disassociated_type"] == "octahedron":
                    self.current_branch_count = self.min_branch_count
                self.disassociated_type = kwargs["disassociated_type"]
                
                self.branch = self.disassociated_branch
            
            case "constant":
                assert "starting_branch_count" in kwargs.keys(), self._handle_bad_args_("branch_method", branch_method, "starting_branch_count")
                self.current_branch_count = kwargs["starting_branch_count"]
                del kwargs["starting_branch_count"]

                self.branch = self.constant_branch
            
            case "test":
                self.branch = self.test_branch
            
            case _:
                raise ValueError("incorrect value passed to branch_method")

        # Set correct workspace_width function in self.workspace_width_method
        match workspace_width_method:
            
            case "constant":
                if workspace_width is None:
                    raise ValueError("workspace_width must be defined for constant workspace width method")
                self.get_workspace_width = self.ww_constant

            case "decrease":
                if workspace_width is None:
                    raise ValueError("workspace_width must be defined for constant workspace width method")                
                self.get_workspace_width = self.ww_decrease

            case "increase":
                if workspace_width is None:
                    raise ValueError("workspace_width must be defined for constant workspace width method")                
                self.get_workspace_width = self.ww_increase
            
            case _:
                raise ValueError("incorrect value passed to workspace_width_method")            
            
        #---------------------------------------------------------------------------------------------------------------    
        # Set the idx value (changes depending on environment/wrapper) of the x and y observations and next_observations
        self.x_obs_idx = kwargs["x_obs_idx"]
        self.y_obs_idx = kwargs["y_obs_idx"]

        # Set initial fractal config values
        self.timestep = 0
        self.current_depth = 0
        self.branch_index = [workspace_width/2]     # TODO: is this a correct initialization?
        self.start_num = kwargs["start_num"]        # Used for fractal contractions

        ## TODO: this is done in insert. Are we repeating work?
        # Create branch index to control how to set (x,y) offsets of different branches
        self.branch_index = np.empty(self.current_branch_count, dtype=np.float32)

        ## Set spacing for transformations (see more: https://www.notion.so/Fractal-Symmetry-Characterization-225cd3402f1a80948509ec86f0b6ee5e?source=copy_link#22bcd3402f1a8054b97ff0ab387923f7)
        # Set a constant value useful in the computation a standard offset between branches
        constant = self.workspace_width/(2 * self.current_branch_count)

        # Compute the translation offsets from left edge according to branch index:
        for i in range(0, self.current_branch_count):
            self.branch_index[i] = (2 * i + 1) * constant

        ## Warn about unused kwargs
        for k in kwargs.keys():
            print(f"\033[33mWARNING \033[0m argument \"{k}\" not used")
        
        # TODO
        #   Create more methods with tests
        # 
        # Future considerations:
        #   when dealing with x and y for transforms, consider two versions:
        #       radial trees at different angles can have strong fractal symmetry built in but will be more difficult to evenly space lowest level transforms
        #       grid-based will inherently evenly space lowest transforms, but fractal symmetry will be restricted to 9 branching_factor (original plus 8) in order to conform

        # Init replay buffer class
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
        )

    # Missing arguments
    def _handle_bad_args_(self, type: str, method: str, arg: str) :
        return f"\033[31mERROR: \033[0m{arg} must be defined for {type} \"{method}\""
    

    # Update (x,y) observations with translational transformation
    def transform(self, data_dict: DatasetDict, transform: np.array):
        #   return data_dict with positional arguments += translation
        data_dict["observations"] += transform
        data_dict["next_observations"] += transform
        
    #--- BRANCH METHODS ---
    # FOR TESTING ONLY
    def test_branch(self):
        self.current_depth += 1
        return 1
    
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
        # return 
        b = self.branching_factor  
        d = self.current_depth
        top_b = self.start_num

        return int( top_b/( b**(d-1) ))
    
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
            return int((self.max_branch_count - self.min_branch_count)/(self.num_depth_sectors/2) * np.abs(self.current_depth - (self.num_depth_sectors/2)) + self.min_branch_count)
        elif self.disassociated_type == "octahedron":
            return int((self.min_branch_count - self.max_branch_count)/(self.num_depth_sectors/2) * np.abs(self.current_depth - (self.num_depth_sectors/2)) + self.max_branch_count)
        
    # REQUIRES TESTING
    # def linear_branch(self):
    #     # return a new number of branches = branches_count + n
    #     return self.current_branch_count + self.branch_count_rate_of_change
    

    #---- SPLIT METHODS ---
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
    
    def disassociated_split(self, data_dict: DatasetDict):
        if self.transition_counter >= self.steps_per_depth:
            self.current_depth += 1
            self.transition_counter = 0
            return True
        else:
            self.transition_counter += 1
            return False

    # Workspace Width Methods
    def ww_constant(self):
        return self.workspace_width
    
    def ww_increase(self):
        '''
        Increase workspace width with depth by 5cm. Lower density.
        '''
        return self.workspace_width + self.current_depth*0.05
    
    def ww_decrease(self):
        '''
        Decrease workspace width with depth by 5cm. Higher density.
        '''
        return self.workspace_width - self.current_depth*0.05         

    #---------------------------------------------
    # Now perform fractal transformations. 
    # Currently iterating through x,y loop (slow).
    # TODO: multigpu processing.

    def insert(self, data_dict_not: DatasetDict):

        # Time
        sector_1 = dt.now() if self.debug_time else None
        data_dict = copy.deepcopy(data_dict_not)

        # Update number of branches if needed
        if self.split(data_dict):

            # Get your current branch count
            temp = self.current_branch_count

            # Get new current branch count based on depth
            self.current_branch_count = self.branch()

            # When we enter into a new depth and new branch count update the transformations
            if temp != self.current_branch_count:

                ## Set spacing for transformations (see more: https://www.notion.so/Fractal-Symmetry-Characterization-225cd3402f1a80948509ec86f0b6ee5e?source=copy_link#22bcd3402f1a8054b97ff0ab387923f7)
                self.branch_index = np.empty(self.current_branch_count, dtype=np.float32)
                constant = self.get_workspace_width()/(2 * self.current_branch_count)

                for i in range(0, self.current_branch_count):
                    self.branch_index[i] = (2 * i + 1) * constant
        
        # Initialize to extreme x and y
        sector_2 = dt.now() if self.debug_time else None

        x = -self.get_workspace_width()/2
        transform = np.zeros_like(data_dict["observations"])
        transform[self.x_obs_idx] = x
        transform[self.y_obs_idx] = x
        self.transform(data_dict, transform)

        # Transform and insert transitions (multiprocessing in the future)
        sector_3 = dt.now() if self.debug_time else None

        for x in range(0, self.current_branch_count):

            # Set offset in x-direction
            transform[self.x_obs_idx] = self.branch_index[x]

            for y in range(0, self.current_branch_count):
                new_data_dict = copy.deepcopy(data_dict)
                transform[self.y_obs_idx] = self.branch_index[y]
                self.transform(new_data_dict, transform)
                super().insert(new_data_dict)
        
        # Reset current_depth, timestep, and max_traj_length
        sector_4 = dt.now() if self.debug_time else None
        self.timestep += 1
        if data_dict["dones"]:
            self.current_depth = 0
            if self.update_max_traj_length:
                self.max_traj_length = int(self.timestep * self.alpha + self.max_traj_length * (1 - self.alpha))
                self.steps_per_depth = int(np.floor(self.max_traj_length/self.num_depth_sectors)) # does this need to be np.float32?
            self.timestep = 0


        if self.debug_time:
            finish = dt.now()
            print(f"Splits: {(sector_2 - sector_1).total_seconds():.5f} : {(sector_3 - sector_2).total_seconds():.5f} : {(sector_4 - sector_3).total_seconds():.5f} : {(finish - sector_4).total_seconds():.5f}\nLaptime: {(finish - sector_1).total_seconds():.5f}")
