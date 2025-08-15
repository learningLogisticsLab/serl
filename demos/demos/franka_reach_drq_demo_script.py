#!/usr/bin/env python3

import time
from time import sleep, perf_counter
from datetime import datetime
import numpy as np
from absl import app, flags, logging

import gym
from oxe_envlogger.envlogger import AutoOXEEnvLogger

import numpy as np
from absl import app, flags

import os

from typing import Any, Dict, Optional
import pickle as pkl
import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

import franka_sim

flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 1000, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_string("demo_path", None, "Path to the demo data.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

flags.DEFINE_integer('num_episodes', 10, 'Number of episodes to log.')
flags.DEFINE_string('output_dir', 'datasets/',
                    'Path in a filesystem to record trajectories.')
flags.DEFINE_string('env_name', 'CartPole-v1', 'Name of the environment.')
flags.DEFINE_boolean('enable_envlogger', False, 'Enable envlogger.')

FLAGS = flags.FLAGS

#-------------------------------------------------------------------------------------------
## Key Config Variables
#-------------------------------------------------------------------------------------------
# Proportional and derivative control gain for action scaling -- empirically tuned
Kp = 10.0      # Values between 20 and 24 seem to be somewhat stable for Kv = 24
Kv = 10.0 

ACTION_MAX = 10 # Maximum action value for clipping actions
ERROR_THRESHOLD = 0.008 # Note!! When this number is changed, the way rewards are computed in the PandaReachCubeEnv.step() L220 must also be changed such that done=True only at the end of a successfull run.

# Number of demonstration episodes to generate
NUM_DEMOS = 20

# Robot configuration
robot = 'franka'    # Robot type used in the environment, can be 'franka' or 'fetch'
task  = 'reach'     # Task type used in the environment, can be 'reach' or 'pick-and-place'

# Debug mode for rendering and visualization
DEBUG = False

if DEBUG:
    _render_mode = 'human'  # Render mode for the environment, can be 'human' or 'rgb_array'
else:
    _render_mode = 'rgb_array'  # Use 'rgb_array' for automated testing without GUI

##############################################################################
def set_front_cam_view(env):
    """
    Set the camera view to a front-facing perspective for better visualization.
    
    Args:
        env: The environment instance containing the viewer.
    
    Returns:
        viewer: The viewer with updated camera settings.
    """
    viewer = env.unwrapped._viewer.viewer  # Access the viewer from the environment
    
    if hasattr(viewer, 'cam'):
        viewer.cam.lookat[:] = [0, 0, 0.1]   # Center of robot (adjust as needed)
        viewer.cam.distance = 3.0            # Camera distance
        viewer.cam.azimuth = 135             # 0 = right, 90 = front, 180 = left
        viewer.cam.elevation = -30           # Negative = above, positive = below

        # Hide menu
        viewer._hide_overlay = True
    
    return viewer

##############################################################################
def compute_error(object_pos, current_pos, prev_error, dt):
    """Compute the error and its derivative between the object position and the current end-effector position.
    Args:
        object_pos (np.ndarray): The position of the object in the environment.
        current_pos (np.ndarray): The current position of the end-effector.
        prev_error (np.ndarray): The previous error value for derivative calculation.
        dt (float): Time step for derivative calculation.
        
    Returns:
        error (np.ndarray): The current error vector between the object and end-effector positions.
        derror (np.ndarray): The derivative of the error vector.
    """
    error = object_pos - current_pos  # Calculate the error vector
    derror = (error - prev_error) / dt  # Calculate the derivative of the error vector 

    prev_error = error.copy()  # Update previous error for next iteration
    return error, derror


def demo(env, lastObs):
    """
    Executes a scripted reach sequence using a hierarchical approach.
    
    Implements 1-phase control strategy:
    1. Approach: Move gripper above object (3cm offset)

    Store observations, actions, and info in global lists for later replay buffer inclusion.

    Gripper:
    - The gripper in Mujoco ranges from a value of 0 to 0.4, where 0 is fully open and 0.4 is fully closed.
    
    Args:
        env: Gymnasium environment instance
        lastObs: Flattened observations set as object_pos, gripper_pos, panda/tcp_pos, panda/tcp_vel
            - observations:
                object_pos[0:3],
                gripper_pos[3]
                panda/tcp_pos[4:7],
                panda/tcp_vel[7:10]

    Returns:
    """

    ## Init goal, current_pos, and object position from last observation
    object_pos       = np.zeros(3, dtype=np.float32)
    current_pos      = np.zeros(3, dtype=np.float32)
    gripper_pos      = np.zeros(1, dtype=np.float32)
    object_rel_pos   = np.zeros(3, dtype=np.float32)


    # Initialize (single) episode data collection
    # episodeObs  = []        # Observations for this episode  
    # episodeAcs  = []        # Actions for this episode
    # episodeRews = []        # Rewards for this episode
    # episodeInfo = []        # Info for this episode
    # episodeTerminated = []  # Terminated flags for this episode
    # episodeTruncated = []   # Truncated flags for this episode
    # episodeDones = []       # Done flags (terminated or truncated) for this episode
    
    # Dictionary to store episode data
    # episode_data = {
    #     "observations": episodeObs,
    #     "actions": episodeAcs,
    #     "rewards": episodeRews,
    #     "infos": episodeInfo,
    #     "terminateds": episodeTerminated,
    #     "truncateds": episodeTruncated,
    #     "dones": episodeDones
    # }   

    # close gripper
    fgr_pos = 0

    # Error thresholds
    error_threshold = ERROR_THRESHOLD  # Threshold for stopping condition (Xmm)

    finger_delta_fast = 0.05    # Action delta for fingers 5cm per step (will get clipped by controller)... more of a scalar. 
    finger_delta_slow = 0.005   # Franka has a range from 0 to 4cm per finger

    ## Extract data   
    object_pos = lastObs[opi[0]:opi[1]]  # block pos
    current_pos = lastObs[rpi[0]:rpi[1]] # panda/tcp_pos

    # Relative position between end-effector and object
    dt = env.unwrapped.model.opt.timestep  # Mujoco time step
    prev_error = np.zeros_like(object_pos)
    error, derror = compute_error(object_pos, current_pos, prev_error, dt)    

    time_step = 0  # Track total time_steps in episode
    episodeObs.append(lastObs) # Store initial observation

    # Initialize previous time for dt calculation
    prev_time = perf_counter()  # Start time for dt calculation
    
    # Phase 1: Reach
    # Terminate when distance to above-object position < error_threshold
    print(f"----------------------------------------------- Phase 1: Reach -----------------------------------------------")
    while np.linalg.norm(error) >= error_threshold and time_step <= env.spec.max_episode_steps:
        env.render()  # Visual feedback
        
        # Record current time and compute dt
        curr_time = perf_counter()
        dt = curr_time - prev_time
        prev_time = curr_time

        # Initialize action vector [x, y, z]
        action = np.array([0., 0., 0.])
        
        # Proportional control with gain of 6
        action[:3] = error * Kp + derror * Kv
        prev_error = error.copy()  # Update previous error for next iteration
        
        # Clip action to prevent excessive movements
        action = np.clip(action/ACTION_MAX, -0.1, 0.1)  #

        # Keep gripper closed -- no need. only 3 dimensions of control
        #action[ len(action)-1 ] = -finger_delta_fast # Maintain gripper closed.
        
        # Unpack new Gymnasium step API
        # new_obs, reward, terminated, truncated, info = env.step(action)
        # done = terminated or truncated
        
        # Store episode data
        #store_transition_data(episode_data, new_obs, reward, action, info, terminated, truncated, done)

        # Update and print state information
        object_pos,gripper_pos,cur_pos,cur_vel = update_state_info(episode_data, time_step, dt, error,reward)

        # Update error for next iteration
        # error, derror = compute_error(object_pos, cur_pos, prev_error, dt)

       # Update time step
        time_step += 1

        # Sleep
        #if DEBUG:
        sleep(0.25)  # Activated when DEBUG is True for better visualization.        

    # Deactivate weld constraint after successful pick -- franka_sim env does not have weld like franka_mujoco env.
    # if weld_flag:
    #     deactivate_weld(env, constraint_name="grasp_weld")

    # Break out of the loop to start a new episode
    return action
        
    # # If we reach here, the episode was not successful
    # if weld_flag:
    #     print("Failed to transport object to goal position. Deactivating weld.")
    #     deactivate_weld(env, constraint_name="grasp_weld")
    
##############################################################################
def main(unused_argv):
    logging.info(f'Creating gym environment...')

    env = gym.make(FLAGS.env_name)

    if FLAGS.env == "PandaPickCube-v0":
        env = gym.wrappers.FlattenObservation(env)
    if FLAGS.env == "PandaPickCubeVision-v0":
        env = SERLObsWrapper(env)
        #env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    logging.info(f'Done creating {FLAGS.env_name} environment.')


    if FLAGS.enable_envlogger:
        env = AutoOXEEnvLogger(
            env=env,
            dataset_name=FLAGS.env_name,
            directory=FLAGS.output_dir,
        )

    logging.info('Training an agent for %r episodes...', FLAGS.num_episodes)

    for i in range(FLAGS.num_episodes):

        # example to log custom metadata during new episode
        if FLAGS.enable_envlogger:
            env.set_episode_metadata({
                "language_embedding": np.random.random((5,)).astype(np.float32)
            })
            env.set_step_metadata({"timestamp": time.time()})

        logging.info('episode %r', i)
        env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()

            # example to log custom step metadata
            if FLAGS.enable_envlogger:
                env.set_step_metadata({"timestamp": time.time()})

            return_step = env.step(action)

            # NOTE: to handle gym.Env.step() return value change in gym 0.26
            if len(return_step) == 5:
                obs, reward, terminated, truncated, info = return_step
            else:
                obs, reward, terminated, info = return_step
                truncated = False

    logging.info(
        'Done training a random agent for %r episodes.', FLAGS.num_episodes)


if __name__ == '__main__':
    app.run(main)