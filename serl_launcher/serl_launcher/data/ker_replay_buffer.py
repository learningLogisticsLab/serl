import copy

import gym
import numpy as np
from serl_launcher.data.dataset import DatasetDict, _sample
from serl_launcher.data.replay_buffer import ReplayBuffer
from transforms3d.euler import euler2mat, euler2quat, quat2euler
from flax.core import frozen_dict

SIMULATION = True

class KerReplayBuffer(ReplayBuffer):
    """
    Class inherits from replay buffer class in order to use KER
    to augment data using reflectional symmetries 
    """
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        workspace_width: int,
        n_KER: int,
        max_z_theta: float,
        z_theta_list: np.array = []
    ):
        self.workspace_width = workspace_width # Total 1D distance for transform calculations
        self.n_KER = n_KER # Number of reflectional planes to generate. Number of new traj = n_ker - 1
        self.max_z_theta = max_z_theta # Max theta possible for generating reflectional planes
        self.z_theta_list = z_theta_list

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
        rot_z_theta = euler2mat(0, 0, z_theta)
        inv_rot_z_theta = euler2mat(0, 0, -z_theta)

        # Determine which state element the param is (eg whether it is an obs or an action)
        param_len = len(param)

        # transform param appropriately
        if SIMULATION: # franka_sim
            if param_len == 4:  #action
                o_act = param[0:3]
                s_act = self.linear_vector_symmetric_with_rot_plane(o_act, rot_z_theta, inv_rot_z_theta)
                param[0:3] =  s_act

            elif param_len == 10:     # observation
                # pos
                o_pos = param[0:3]
                s_pos = self.linear_vector_symmetric_with_rot_plane(o_pos, rot_z_theta, inv_rot_z_theta)
                param[0:3] =  s_pos
                # vel
                o_vel = param[3:6]
                s_vel = self.linear_vector_symmetric_with_rot_plane(o_vel, rot_z_theta, inv_rot_z_theta)
                param[3:6] =  s_vel
                # obj_pos
                o_obj_pos = param[7:10]
                s_obj_pos = self.linear_vector_symmetric_with_rot_plane(o_obj_pos, rot_z_theta, inv_rot_z_theta)
                param[7:10] =  s_obj_pos
        # else: # Real robot
        #     if param_len == 7:  # action
        #         # xyz
        #         o_act_pos = param[0:3]
        #         s_act_pos = self.linear_vector_symmetric_with_rot_plane(o_act_pos, rot_z_theta, inv_rot_z_theta)
        #         param[0:3] =  s_act_pos
        #         # quat
        #         o_act_quat = param[3:7]
        #         s_act_quat = self.reflect_orientation_with_rot_plane(o_act_quat, rot_z_theta, inv_rot_z_theta)
        #         param[3:7] = s_act_quat



        #     elif param_len == 20: # observation
                

        return param.copy()
    
    def linear_vector_symmetric_with_rot_plane(self, o_data, rot_z_theta, inv_rot_z_theta):
        # Point 'a' position = v_l_a
        o_data_hat = np.dot(inv_rot_z_theta,o_data)
        o_data_hat[1] = -o_data_hat[1]
        s_data =  np.dot(rot_z_theta,o_data_hat)
        return s_data.copy()
    
    # def reflect_orientation_with_rot_plane(self, o_data, theta):
    #     # reflects orientation about plane given by theta

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
    
    def insert(self, ka_episodes_set, base_data_dict):
        ''' Reformats transformed episodes and inserts them into the replay buffer
        '''
        data_dict = copy.deepcopy(base_data_dict)
        for episode in ka_episodes_set:
            # Overwrite initial data_dict values for observations, next_observations, and actions with transformed versions
            data_dict['observations'] = episode[0]
            data_dict['next_observations'] = episode[1]
            data_dict['actions'] = episode[2]
            # Insert into replay buffer using parent class method
            super().insert(data_dict)