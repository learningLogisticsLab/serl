import copy

import gym
import numpy as np
from serl_launcher.data.dataset import DatasetDict, _sample
from serl_launcher.data.replay_buffer import ReplayBuffer

from transforms3d.euler import euler2mat, mat2euler, quat2euler, euler2quat
from flax.core import frozen_dict

class KerReplayBuffer(ReplayBuffer):
    """
    Class inherits from replay buffer class in order to use KER
    to augment data using reflectional symmetries 
    """
    def __init__(
        self,
        #env: gym.Env,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        n_KER: int,
        max_z_theta: float,
        z_theta_list: np.array = [] # Eventually get rid of this as we will want complete randomness after testing
    ):
        if z_theta_list is None:
            z_theta_list = []
        self.n_KER = n_KER # Number of reflectional planes to generate. Number of new traj = n_ker - 1
        self.max_z_theta = max_z_theta # Max theta possible for generating reflectional planes
        self.z_theta_list = z_theta_list

        ## Resolve Circular import isuses???

        #env_name = env.spec.id
        # self.ker_buffer = KerReplayBuffer(
        #     observation_space=observation_space,
        #     action_space=action_space,
        #     capacity=capacity,
        #     n_KER=kwargs.get('n_KER', 4),  # Default to 4 if not provided
        #     max_z_theta=kwargs.get('max_z_theta', np.pi/4)  # Default to 45 degrees if not provided
        # )

        #if env_name.lower() == 'pandapickcube-v0' or env_name.lower() == 'pandareachcube-v0':
        # self.env_type = 'simulation'
        # self.pos_idx = np.array([0,1,2])
        # self.vel_idx = np.array([4,5,6])
        # self.bp_idx = np.array([7,8,9]) # Block position
        # self.acts_idx = np.array([0,1,2])
        # self.translational_obs = np.array([self.pos_idx, self.vel_idx, self.bp_idx])
        # self.rotational_obs = np.array([])
        # self.action_space_dim = action_space.shape[0]

        # # elif env_name.lower() == 'frankaenv-vision-v0' or env_name.lower() == 'frankapeginsert-vision-v0' or env_name.lower() == 'frankapcbinsert-vision-v0' or env_name.lower() == 'frankacableroute-vision-v0' or env_name.lower() == 'frankabinrelocation-vision-v0':
        #     self.env_type = 'physical'
        #     self.pos_idx = np.array([0,3])
        #     self.quat_idx = np.array([3,7])
        #     self.lin_vel_idx = np.array([7,10])
        #     self.ang_vel_idx = np.array([10,13])
        #     self.gp_idx = np.array([13])
        #     self.force_idx = np.array([14,17])
        #     self.torque_idx = np.array([17,20])

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity
        )
    
    def linear_vector_symmetric_with_rot_plane(self, data_dict, rot_z_theta, inv_rot_z_theta):
        # Point 'a' position = v_l_a
        for obs in self.translational_obs:
            o_data = data_dict["observations"][obs]
            o_data_hat = np.dot(inv_rot_z_theta,o_data)
            o_data_hat[1] = -o_data_hat[1]
            s_data =  np.dot(rot_z_theta,o_data_hat)
            data_dict["observations"][obs] = s_data

            o_data = data_dict["next_observations"][obs]
            o_data_hat = np.dot(inv_rot_z_theta,o_data)
            o_data_hat[1] = -o_data_hat[1]
            s_data =  np.dot(rot_z_theta,o_data_hat)
            data_dict["next_observations"][obs] = s_data
            
        # TODO fix spaghetti
        o_data = data_dict["actions"][self.acts_idx]
        o_data_hat = np.dot(inv_rot_z_theta,o_data)
        o_data_hat[1] = -o_data_hat[1]
        s_data =  np.dot(rot_z_theta,o_data_hat)
        data_dict["actions"][self.acts_idx] = s_data
        return

        
    
    # def reflect_orientation_with_rot_plane(self, o_data, rot_z_theta, inv_rot_z_theta):
    #     # reflects orientation about plane given by theta
    #     o_mat = euler2mat(o_data[0],o_data[1],o_data[2], axes='rxyz')
    #     o_mat_hat = np.matmul(o_mat, inv_rot_z_theta)
    #     o_euler = np.array(mat2euler(o_mat_hat, axes='rxyz'), dtype=np.float32)
    #     o_euler[0] = -o_euler[0]
    #     o_euler[2] = -o_euler[2]
    #     s_mat = euler2mat(o_euler[0],o_euler[1],o_euler[2], axes='rxyz')
    #     s_mat_hat = np.matmul(s_mat, rot_z_theta)
    #     s_euler = np.array(mat2euler(s_mat_hat, axes='rxyz'),dtype=np.float32)
    #     return s_euler.copy()


    def insert(self, base_data_dict):
        ''' Reformats transformed episodes and inserts them into the replay buffer
        '''
        data_dict = copy.deepcopy(base_data_dict)

        # generate angles for reflection
        if not self.z_theta_list:
            self.z_theta_list = []
            for _ in range(self.n_KER - 1):
                self.z_theta_list.append(np.random.uniform(0, self.max_z_theta))

        y_axis_flip = euler2mat(0, 0, np.pi, axes="rxyz")

        # insert original + reflected transitions
        super().insert(data_dict)
        new_data_dict = copy.deepcopy(data_dict)

        self.linear_vector_symmetric_with_rot_plane(new_data_dict, y_axis_flip, y_axis_flip)
        # self.reflect_orientation_with_rot_plane()

        super().insert(new_data_dict)

        for z_theta in self.z_theta_list:
            # initial transitions
            rot_z_theta = euler2mat(0, 0, z_theta, axes='rxyz')
            inv_rot_z_theta = euler2mat(0, 0, -z_theta, axes='rxyz')
            new_data_dict = copy.deepcopy(data_dict)
            self.linear_vector_symmetric_with_rot_plane(new_data_dict, rot_z_theta, inv_rot_z_theta)
            # self.reflect_orientation_with_rot_plane()

            super().insert(new_data_dict)

            # flipped across y-axis
            self.linear_vector_symmetric_with_rot_plane(new_data_dict, y_axis_flip, y_axis_flip)
            # self.reflect_orientation_with_rot_plane()

            super().insert(new_data_dict)