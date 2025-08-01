# Generic support
import copy
from datetime import datetime as dt

# RL Envs
import gym

# Data
from serl_launcher.data.dataset import DatasetDict
from serl_launcher.data.replay_buffer import ReplayBuffer

# Optimization of matrix operations
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


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
        self.current_branch_count = 0
        self.num_transforms = 0

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
    
    def compute_transformation(self, data_dict, num_transforms):
        '''
        
        '''

        # Generate observation matrices
        o_m, no_m = self.generate_obs_mats(data_dict, num_transforms)

        # Transform o_m and no_m using fractal equation (use lax.scan inside for incremental change)
        del_o_m = lax.scan( self.compute_delta, o_m )
        del_no_m = lax.scan( self.compute_deltas, no_m)

        return del_o_m,del_no_m


    def compute_deltas(self,mat,type='grid'): # add type as jnp array
        '''
        Given a type of fractal framework perform the operation
        '''

        if type == 'grid':

            # Iterate through matrix cols
            idx = jnp.arange(self.num_transforms)

            # Convert idx into arrays (i,j) that tell us the x,y order for deltas
            i,j = jnp.divmod( idx, self.current_branch_count )

            # 1. Get the boundary of the workspace environment
            edge = self.get_workspace_width()/2.

            # 2. Get array of translation deltas: ( (2i+1)/2b ) * ww
            constant = self.get_workspace_width() / (2. * self.current_branch_count)
            x_offset = ( (2 * i) + 1) * constant
            y_offset = ( (2 * j) + 1) * constant

            # 3. Enter for block and robot x,y positions for all transformations
            #             from jax import lax

            # m_updated = lax.scatter(m,
            #     indices=r[:, None],
            #     updates=new_rows,
            #     dimension_numbers=lax.ScatterDimensionNumbers(
            #         update_window_dims=(1,),
            #         inserted_window_dims=(0,),
            #         scatter_dims_to_operand_dims=(0,)
            #     ),
            #     indices_are_sorted=False,
            #     unique_indices=True
)


        else:
            raise TypeError('transform_matrix: no correct fractal framework type was given...')
        
    def generate_obs_mats(self, data_dict: DatasetDict, num_transforms: int, type='grid'):
        '''
        Generate observations matrices, whose cols repeat according to the correct number of transformations

        Args:
        - data_dict (DatasetDict): 
        - num_transforms (int): computes the total number of transformations for the grid stategy
        - type (str): grid or radial fractal framework

        Return:
        - o_m: observation matrix
        - no_m: next observation matrix
        '''
        
        # Convert observations to a vector
        o_col = jnp.array(data_dict["observations"], type=jnp.float32)
        no_col = jnp.array(data_dict["next_observations"], type=jnp.float32)

        # Expand observations column vector to a matrix using tile
        o_m = jnp.tile(o_col, (1,num_transforms))
        no_m = jnp.tile(no_col, (1,num_transforms))

        return o_m, no_m

    def insert_mat_replay_buffer(self, del_o, del_no, new_data_dict: DatasetDict):
        # Convert back to dict by replacing data_dict's observations with new ones by the columns and insert into replay buffer one-by-one. 
        # TODO: Is there a batch_insert?

        # Is this the fast way to create these dicts from matrix cols?
        for col in range( del_o.shape[1] ):
            new_data_dict['observations'] = np.array( del_o[:, col],type=np.float32)
            new_data_dict['next_observations'] = np.array( del_no[:, col],type=np.float32)

            # Insert each col's new info
            super().insert(new_data_dict)        

    def insert(self, data_dict: DatasetDict):
        '''
        
        Note:
        If observations for all time-steps are available, then a batch_transform 
        '''

        # Time
        sector_1 = dt.now() if self.debug_time else None

        # Update number of branches if needed
        if self.split(data_dict):

            # Get the current branch count
            temp = self.current_branch_count

            # Get new current branch count based on depth
            self.current_branch_count = self.branch()

            # When we enter into a new depth and new branch count update the transformations
            if temp != self.current_branch_count:

                # Get the number of transformations
                self.num_transforms = self.current_branch_count**2

                # Compute transformation deltas for observations
                self.delta_o_m, self.delta_no_m = self.compute_transformation(data_dict, self.num_transforms)

                # Add observations with deltas
                transf_o_m = o_m + self.delta_o_m
                transf_no_m = no_m + self.delta_no_m

                insert_mat_replay_buffer(transf_o_m, transf_no_m)

        #else??

        # Generate observation matrices with same
        o_m, no_m = self.generate_obs_mats(data_dict, self.num_transforms)

        # Add observations with deltas
        transf_o_m = o_m + self.delta_o_m
        transf_no_m = no_m + self.delta_no_m    

        # Insert to replay buffer
        insert_mat_replay_buffer(transf_o_m, transf_no_m)    


        # Reset current_depth, timestep, and max_traj_length
        sector_4 = dt.now() if self.debug_time else None

        self.timestep += 1
        if data_dict["dones"]:
            self.current_depth = 0
            if self.update_max_traj_length:
                self.max_traj_length = int(self.timestep * self.alpha + self.max_traj_length * (1 - self.alpha))
            self.timestep = 0

        if self.debug_time:
            finish = dt.now()
            print(f"Splits: {(sector_2 - sector_1).total_seconds():.5f} : {(sector_3 - sector_2).total_seconds():.5f} : {(sector_4 - sector_3).total_seconds():.5f} : {(finish - sector_4).total_seconds():.5f}\nLaptime: {(finish - sector_1).total_seconds():.5f}")