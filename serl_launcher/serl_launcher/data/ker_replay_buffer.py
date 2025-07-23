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
        self.n_KER = n_KER # Number of reflectional planes to generate. Number of new traj = n_ker - 1
        self.max_z_theta = max_z_theta # Max theta possible for generating reflectional planes
        self.z_theta_list = z_theta_list
        #env_name = env.spec.id




        #if env_name.lower() == 'pandapickcube-v0' or env_name.lower() == 'pandareachcube-v0':
        self.env_type = 'simulation'
        self.pos_idx = np.array([0,3])
        self.vel_idx = np.array([4,7])
        self.bp_idx = np.array([7,10]) # Block position
        self.action_space_dim = action_space.shape[0]
        self.obs_transform_idxs = [self.pos_idx, self.vel_idx, self.bp_idx]

        # elif env_name.lower() == 'frankaenv-vision-v0' or env_name.lower() == 'frankapeginsert-vision-v0' or env_name.lower() == 'frankapcbinsert-vision-v0' or env_name.lower() == 'frankacableroute-vision-v0' or env_name.lower() == 'frankabinrelocation-vision-v0':
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
    
    def y_ker(self,param):
        return self.kaleidoscope_robot(param, 0)
    
    def kaleidoscope_robot(self, param, z_theta, sym_axis = 'y_axis', sym_method = 'y_ker'):
        ''' Will compute transformations of (s,a,r,s') according to transportation.
        '''
        # compute the rotation transformation & its inverse.
        rot_z_theta = euler2mat(0, 0, z_theta, axes="rxyz")
        inv_rot_z_theta = euler2mat(0, 0, -z_theta, axes="rxyz")

        # Determine which state element the param is (eg whether it is an obs or an action)
        param_len = len(param)

        # transform param appropriately
        if self.env_type == 'simulation':   # franka_sim
            if param_len == 3 or param_len == 4:  #action
                o_act = param[0:3]
                s_act = self.linear_vector_symmetric_with_rot_plane(o_act, rot_z_theta, inv_rot_z_theta)
                param[0:3] =  s_act

            elif param_len == 10:     # observation
                for bounds in self.obs_transform_idxs:
                    self.perform_transformation(param, bounds[0], bounds[1], rot_z_theta, inv_rot_z_theta)
                # # tcp_pos
                # o_pos = param[self.pos_idx[0]:self.pos_idx[1]]
                # s_pos = self.linear_vector_symmetric_with_rot_plane(o_pos, rot_z_theta, inv_rot_z_theta)
                # param[self.pos_idx[0]:self.pos_idx[1]] =  s_pos
                # # tcp_vel
                # o_vel = param[self.vel_idx[0]:self.vel_idx[1]]
                # s_vel = self.linear_vector_symmetric_with_rot_plane(o_vel, rot_z_theta, inv_rot_z_theta)
                # param[self.vel_idx[0]:self.vel_idx[1]] =  s_vel
                # # obj_pos
                # o_obj_pos = param[self.bp_idx[0]:self.bp_idx[1]]
                # s_obj_pos = self.linear_vector_symmetric_with_rot_plane(o_obj_pos, rot_z_theta, inv_rot_z_theta)
                # param[self.bp_idx[0]:self.bp_idx[1]] =  s_obj_pos


        elif self.env_type == 'physical':   # Real robot
            if param_len == 7:  # action
                # xyz
                o_act_pos = param[0:3]
                s_act_pos = self.linear_vector_symmetric_with_rot_plane(o_act_pos, rot_z_theta, inv_rot_z_theta)
                param[0:3] =  s_act_pos
                # quat
                o_act_quat = param[3:7]
                o_act_euler = quat2euler(o_act_quat, axes='rxyz')
                s_act_euler = self.reflect_orientation_with_rot_plane(o_act_euler, rot_z_theta, inv_rot_z_theta)
                s_act_quat = euler2quat(s_act_euler[0],s_act_euler[1],s_act_euler[2], axes='rxyz')
                param[3:7] = s_act_quat

            # elif param_len == 20: # observation
            #     # tcp_pos
            #     o_pos = param[self.pos_idx[0]:self.pos_idx[1]]
            #     s_pos = self.linear_vector_symmetric_with_rot_plane(o_pos, rot_z_theta, inv_rot_z_theta)
            #     param[self.pos_idx[0]:self.pos_idx[1]] =  s_pos
            #     # tcp_quat
            #     o_quat = param[3:7]
            #     o_euler = quat2euler(o_quat, axes='rxyz')
            #     s_euler = self.reflect_orientation_with_rot_plane(o_euler, rot_z_theta, inv_rot_z_theta)
            #     s_quat = euler2quat(s_euler[0],s_euler[1],s_euler[2], axes='rxyz')
            #     param[3:7] = s_quat
            #     # tcp_linear_vel
            #     o_lin_vel = param[self.lin_vel_idx[0]:self.lin_vel_idx[1]]
            #     s_lin_vel = self.linear_vector_symmetric_with_rot_plane(o_lin_vel, rot_z_theta, inv_rot_z_theta)
            #     param[self.lin_vel_idx[0]:self.lin_vel_idx[1]] =  s_lin_vel
            #     # tcp_ang_vel
            #     o_ang_vel = param[self.lin_ang_idx[0]:self.lin_ang_idx[1]]
            #     s_ang_vel = self.reflect_orientation_with_rot_plane(o_ang_vel, rot_z_theta, inv_rot_z_theta)
            #     param[self.ang_vel_idx[0]:self.ang_vel_idx[1]] =  s_ang_vel
            #     # tcp_force
            #     o_force = param[self.force_idx[0]:self.force_idx[1]]
            #     s_force = self.linear_vector_symmetric_with_rot_plane(o_force, rot_z_theta, inv_rot_z_theta)
            #     param[self.force_idx[0]:self.force_idx[1]] =  s_force
            #     # tcp_torque
            #     o_torque = param[self.torque_idx[0]:self.torque_idx[1]]
            #     s_torque = self.reflect_orientation_with_rot_plane(o_torque, rot_z_theta, inv_rot_z_theta)
            #     param[self.torque_idx[0]:self.torque_idx[1]] =  s_torque


        return param.copy()
    
    def perform_transformation(self, param, lower_idx, upper_idx, rot_z_theta, inv_rot_z_theta):
        ### Does not work for physical robot!!
        original_values = param[lower_idx:upper_idx]
        if len(original_values) == 3:
            transformed_values = self.linear_vector_symmetric_with_rot_plane(original_values, rot_z_theta, inv_rot_z_theta)
        # elif len(original_values) == 4:
        #     transformed_values = self.reflect_orientation_with_rot_plane(original_values, rot_z_theta, inv_rot_z_theta)
        param[lower_idx:upper_idx] = transformed_values


    def linear_vector_symmetric_with_rot_plane(self, o_data, rot_z_theta, inv_rot_z_theta):
        # Point 'a' position = v_l_a
        o_data_hat = np.dot(inv_rot_z_theta,o_data)
        o_data_hat[1] = -o_data_hat[1]
        s_data =  np.dot(rot_z_theta,o_data_hat)
        return s_data.copy()
    
    def reflect_orientation_with_rot_plane(self, o_data, rot_z_theta, inv_rot_z_theta):
        # reflects orientation about plane given by theta
        o_mat = euler2mat(o_data[0],o_data[1],o_data[2], axes='rxyz')
        o_mat_hat = np.matmul(o_mat, inv_rot_z_theta)
        o_euler = np.array(mat2euler(o_mat_hat, axes='rxyz'), dtype=np.float32)
        o_euler[0] = -o_euler[0]
        o_euler[2] = -o_euler[2]
        s_mat = euler2mat(o_euler[0],o_euler[1],o_euler[2], axes='rxyz')
        s_mat_hat = np.matmul(s_mat, rot_z_theta)
        s_euler = np.array(mat2euler(s_mat_hat, axes='rxyz'),dtype=np.float32)
        return s_euler.copy()


    def ker_process(self,data_dict):
        ''' Will do invariant transform augmentation. Augments time-steps by 2nker - 1 + nger
        '''
        # ---------------------------linear symmetry------------------------------------------------
        keys = ['observations', 'next_observations', 'actions', 'rewards', 'masks', 'dones']
        
        # Extract s a s'
        obs = data_dict[keys[0]]
        next_obs = data_dict[keys[1]]
        acts = data_dict[keys[2]]

        ka_episodes_set = []
        ka_episodes_set.append([obs, next_obs, acts]) # Add next_obs later

        # If the user has not passed in a specific z_theta_list, fill it randomly
        if self.z_theta_list == []:
            # One symmetry will be done in the y ker, so here n_KER need to minus 1
            for _ in range(self.n_KER-1):
                z_theta = np.random.uniform(0, self.max_z_theta)
                self.z_theta_list.append(z_theta)

        ka_episodes_tem = []
        for z_theta in self.z_theta_list:

            for [o_obs, o_next_obs, o_acts] in ka_episodes_set:
                # Symmetric counterparts
                s_ob = self.kaleidoscope_robot(o_obs.copy(),z_theta)
                s_next_ob = self.kaleidoscope_robot(o_next_obs.copy(),z_theta)
                s_act = self.kaleidoscope_robot(o_acts.copy(),z_theta)

                ka_episodes_tem.append([s_ob.copy(), s_next_ob.copy(), s_act.copy()])
        for ka_episode in ka_episodes_tem:
            ka_episodes_set.append(ka_episode)
        # ---------------------------end
        
        #--------------- All datas are symmetrized by the x-axis.
        yker_episode_set = []
        for [o_obs, o_next_obs, o_acts] in ka_episodes_set:
    
            y_ob = self.y_ker(o_obs.copy())
            y_next_ob = self.y_ker(o_next_obs.copy())
            y_act = self.y_ker(o_acts.copy())

            yker_episode_set.append([y_ob.copy(), y_next_ob.copy(), y_act.copy()])

        for yker_episode in yker_episode_set:
            ka_episodes_set.append(yker_episode)

        return ka_episodes_set
        #--------------- end.
    
    def insert(self, base_data_dict, ka_episodes_set):
        ''' Reformats transformed episodes and inserts them into the replay buffer
        '''
        data_dict = copy.deepcopy(base_data_dict)
        #ka_episodes_set = self.ker_process(data_dict)

        for episode in ka_episodes_set:
            # Overwrite initial data_dict values for observations, next_observations, and actions with transformed versions
            data_dict['observations'] = episode[0]
            data_dict['next_observations'] = episode[1]
            data_dict['actions'] = episode[2]
            # Insert into replay buffer using parent class method
            super().insert(data_dict)