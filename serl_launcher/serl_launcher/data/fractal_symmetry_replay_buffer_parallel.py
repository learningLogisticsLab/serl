# Generic support
import copy
from datetime import datetime as dt

# RL Envs
import gym

# Data
from serl_launcher.data.dataset import DatasetDict
from serl_launcher.data.fractal_symmetry_replay_buffer import FractalSymmetryReplayBuffer

# Optimization of matrix operations
import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial

class FractalSymmetryReplayBufferParallel(FractalSymmetryReplayBuffer):
    '''
    Override .insert(), add batch_insert(), override __init__ if needed, etc.
    '''
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
               
        # parallel-specific setup...
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
            branch_method=branch_method,
            split_method=split_method,
            workspace_width=workspace_width,
            workspace_width_method=workspace_width_method,
            kwargs=kwargs,
        )
        
        # Set the total number of transformations to the number of transformations squared for the grid method. May need to change for radial method later.
        self.num_transforms = self.current_branch_count ** 2

    #---------------------------------------------
    @partial(jax.jit, static_argnums=(0, 2)) # states that self and num_transforms are not array like and do not need to be traced and need to be marked as static.
    def compute_transformation(self, data_dict, num_transforms):
        '''
        Add fractal transformations (translations) to matrix of observations in jitted form.
        - data_dict of transitions comes in, a matrix of size (num_transforms,num_transforms) comes out. 
        - compute_deltas will create a matrix of (x,y) transformations and add them to the expanded observations.
        - results for observations and next_observations saved in trans_o_m and trans_no_m.

        Only this parent function needs to be jitted. Helper functions below do not need since they are never called independently.

        Params:
        - data_dict (DatasetDict) - dictionary of translations
        - num_transforms (int) - number of transformations

        Returns:
        - trans_o_m (jnp.ndarray) - this is a (num_transform,num_transform) matrix
        - trans_no_m (jnp.ndarray) - this is a (num_transform,num_transform) matrix

        '''

        # Generate observation matrices
        o_m, no_m = self.generate_obs_mats(data_dict, num_transforms)

        # Transform o_m and no_m using fractal equation (use lax.scan inside for incremental change)
        trans_o_m = self.compute_deltas( o_m )
        trans_no_m = self.compute_deltas( no_m)

        return trans_o_m,trans_no_m

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

            # 2. Get array of translation deltas for x,y offsets: ( (2i+1)/2b ) * ww
            constant_parallel = self.get_workspace_width() / (2. * self.current_branch_count)
            x_offset = ( (2 * i) + 1) * constant_parallel
            y_offset = ( (2 * j) + 1) * constant_parallel

            # 3. Generate the matrix grid of transformations by element-wise sum of deltas on the matrix of env. observations. Use self.x_obs_idx & y_obs_idx.
            # Perform separate updates
            trans_mat = (
                mat
                .at[self.x_obs_idx].add(x_offset)          # add x_offset to each of the two x-rows
                .at[self.y_obs_idx].add(y_offset)          # add y_offset to each of the two y-rows
            )
        else:
            raise TypeError('transform_matrix: no correct fractal framework type was given...')
        
        return trans_mat
        
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
        o_col = jnp.array(data_dict["observations"], dtype=jnp.float32)
        no_col = jnp.array(data_dict["next_observations"], dtype=jnp.float32)

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

            # New depth -> new branch count. Hence update the transformations:
            if temp != self.current_branch_count:

                # Get the number of transformations
                self.num_transforms = self.current_branch_count**2

                # Compute transformation deltas for observations
                sector_2 = dt.now() if self.debug_time else None
                trans_o_m, trans_no_m = self.compute_transformation(data_dict, self.num_transforms)

                # Insert to replay buffer
                sector_3 = dt.now() if self.debug_time else None
                batch_insert(trans_o_m, trans_no_m, data_dict)

        #else??
        # Compute transformation deltas for observations
        sector_2 = dt.now() if self.debug_time else None
        trans_o_m, trans_no_m = self.compute_transformation(data_dict, self.num_transforms)
  
        # Insert as batch into replay buffer
        sector_3 = dt.now() if self.debug_time else None
        batch_insert(trans_o_m, trans_no_m, data_dict)

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

        def batch_insert(self, trans_o_m: jnp.ndarray, trans_no_m: jnp.ndarray, data_dict: DatasetDict) -> None:
            """
            Inserts N data-augmented transitions into self.dataset_dict all at once.

            Args:
            data_dict: original single transition dict with keys
                - "observations"       shape (obs_dim,)
                - "next_observations"  shape (obs_dim,)
                - "actions"            shape (act_dim,)
                - "rewards"            scalar or shape ()
                - "dones"              scalar bool
                - "masks"              scalar or shape ()
            trans_o_m: jnp.ndarray (obs_dim, N) — each column is a new obs
            trans_no_m: jnp.ndarray (obs_dim, N) — each column is new next_obs

            Returns:
            None.  Mutates self.dataset_dict and advances the insert pointer.
            """
        
            # Extract number of transformations directly from the data
            N = trans_o_m.shape[1]

            # Expand observations and next observations
            # Transpose matrix of observations and next_observations as a numpy array: trans_o_m: (obs_dim, N) → we want (N, obs_dim) in NumPy
            obs_batch     = np.array(trans_o_m.T)      # shape (N, obs_dim)
            next_obs_batch= np.array(trans_no_m.T)     # shape (N, obs_dim)

            # Expand actions:
            actions = np.array(data_dict["actions"], dtype=np.float32)
            
            # If shape (act_dim,), broadcast to (N, act_dim)
            actions_batch = np.broadcast_to(actions, (N,) + actions.shape)

            # Rewards/dones/masks batches
            rewards = float(data_dict["rewards"])
            dones   = bool(data_dict["dones"])
            masks   = bool(data_dict["masks"])  # check type

            rewards_batch = np.full((N,), rewards, dtype=np.float32)
            dones_batch   = np.full((N,), dones,   dtype=bool)
            masks_batch   = np.full((N,), masks,   dtype=np.float32)

            batch = {
                "observations":      obs_batch,
                "next_observations": next_obs_batch,
                "actions":           actions_batch,
                "rewards":           rewards_batch,
                "dones":             dones_batch,
                "masks":             masks_batch,
                }
            
            # Create appropriate indeces for circular buffer. Consider overflow:
            idxs = (self._insert_index + np.arange(N)) % self._capacity

            # One‐shot write into each field
            for key, arr in batch.items():  # arr has shape (N, *field_shape)
                self.dataset_dict[key][idxs] = arr

            # Update pointers
            self._insert_index = int(idxs[-1]) + 1
            self._size = min(self._size + N, self._capacity)
            