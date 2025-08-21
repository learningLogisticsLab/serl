#!/usr/bin/env python3

import time
# from time import perf_counter

import numpy as np
from absl import app, flags, logging

import gym
from oxe_envlogger.envlogger import AutoOXEEnvLogger

import numpy as np
from absl import app, flags

import os

import pickle as pkl
import gym

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

import franka_sim

# Teleoperation imports
import sys, select, termios, tty

#-------------------------------------------------------------------------------------------
# Flags
#-------------------------------------------------------------------------------------------
flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Name of environment.")

#flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 200, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")

# flag to indicate if this is a leaner or a actor
#flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
#flags.DEFINE_string("demo_path", None, "Path to the demo data.")

flags.DEFINE_boolean("debug", True, "Debug mode.")  # debug mode will disable wandb logging

flags.DEFINE_string("log_rlds_path", "/data/data/serl/demos/franka_reach_drq_demo_script", "Path to save RLDS logs.")
#flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")
flags.DEFINE_string("output_dir", "/data/data/serl/demos/franka_reach_drq_demo_script",
                    "Directory to save the output data. This is where the RLDS logs will be saved.")
                     
flags.DEFINE_integer("num_demos", 2, "Number of episodes to log.")

flags.DEFINE_boolean("enable_envlogger", True, "Enable envlogger.")

# Keyboard teleoperation
flags.DEFINE_string("teleop_mode", "keyboard", "Teleoperation mode: 'keyboard' or 'spacemouse'.")

FLAGS = flags.FLAGS

#-------------------------------------------------------------------------------------------
## Telop Config Variables
#-------------------------------------------------------------------------------------------
ACTION_MAX = 10 # Maximum action value for clipping actions

# Bind, xyz, gripper vals to keys
moveBindings = {
    'i':(1,0,0,0),
    ',':(-1,0,0,0),
    'j':(0,1,0,0),
    'l':(0,-1,0,0),
    'u':(0,0,0,1),
    'o':(0,0,0,-1),
    'm':(0,0,1,0),
    '.':(0,0,-1,0),
        }
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
        viewer.cam.azimuth = 155             # 0 = right, 90 = front, 180 = left
        viewer.cam.elevation = -30           # Negative = above, positive = below

        # Hide menu
        viewer._hide_overlay = True
    
    return viewer

def getKey(settings):
    """
    Waits briefly for a keypress and returns the pressed key.

    Parameters
    ----------
    settings : list
        Original terminal settings so they can be restored after reading.

    Returns
    -------
    key : str
        The key pressed by the user, or '' (empty string) if no key was pressed.
    """
    # Put terminal into raw mode so keypress is captured instantly
    tty.setraw(sys.stdin.fileno())

    # Wait for human to input action, we will not provide a timeout so it is blocking.
    rlist, _, _ = select.select([sys.stdin], [], []) # select.select(rlist, wlist, xlist[, timeout])

    if rlist:
        # Read exactly one character if a key is pressed
        key = sys.stdin.read(1)
    else:
        key = ''

    # Restore terminal to original settings
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def get_kb_demo_action(speed=0.1):
    """
    Reads keyboard input and maps it to a 3D action vector for robot control.

    The function uses non-blocking keyboard input to allow interactive
    teleoperation. Keys are mapped to directions in Cartesian space:
      - 'i' : +x (forward)
      - ',' : -x (backward)
      - 'j' : +y (left)
      - 'l' : -y (right)
      - 'm' : +z (up)
      - '.' : -z (down)
      - 'k' : stop (zero vector)

    Parameters
    ----------
    speed : float, optional
        Step size for each key press (default 0.2).

    Returns
    -------
    np.ndarray
        Action vector of shape (3,), where each entry corresponds to
        [x, y, z] translation command. Example: [0.2, 0.0, 0.0].
    """
    # Save current terminal settings so we can restore later
    settings = termios.tcgetattr(sys.stdin)

    # Initialize action as a zero vector (no movement)
    action = np.zeros(4, dtype=float)

    try:
        # Capture the pressed key 
        key = getKey(settings)

        if key in moveBindings:
            # Lookup (x, y, z) direction and scale by speed
            dx, dy, dz, g= moveBindings[key]
            action = np.array([dx, dy, dz, g], dtype=float) * speed

        elif key == 'k':
            # 'k' means stop → zero vector
            action = np.zeros(4, dtype=float)

        elif key == '\x03':  # CTRL-C
            raise KeyboardInterrupt
        
    finally:
        # Restore terminal even if something goes wrong
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    # Clip action values to prevent excessive commands
    action = np.clip(action, -ACTION_MAX, ACTION_MAX)

    return action

##############################################################################
def main(unused_argv):
    logging.info(f'Creating gym environment...')

    # Render mode configuration based on debug flag
    if FLAGS.debug:
        _render_mode = 'human'  # Render mode for the environment, can be 'human' or 'rgb_array'
    else:
        _render_mode = 'rgb_array'  # Use 'rgb_array' for automated testing without GUI

    # Create the environment with the specified render mode and wrappers
    env = gym.make(FLAGS.env, render_mode=_render_mode)

    if FLAGS.env == "PandaPickCube-v0":
        env = gym.wrappers.FlattenObservation(env)

    if FLAGS.env == "PandaPickCubeVision-v0":
        env = SERLObsWrapper(env)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    logging.info(f'Done creating {FLAGS.env} environment.')

    # Set the camera view to a front-facing perspective
    if hasattr(env.unwrapped, '_viewer'):
        viewer = set_front_cam_view(env)
        if viewer:
            logging.info('Camera view set to front-facing perspective.')
        else:
            logging.warning('Failed to set camera view. Viewer not available.')

    # If envlogger is enabled, wrap the environment with AutoOXEEnvLogger to log episodes
    if FLAGS.enable_envlogger:
        env = AutoOXEEnvLogger(
            env=env,
            dataset_name=FLAGS.env,
            directory=FLAGS.output_dir,
        )

    logging.info('Training an agent for %r episodes...', FLAGS.num_demos)

    #--- LOOP DEMOS/EPISODES ---
    # Loop through the number of demos specified by the user to record demonstrations
    for i in range(FLAGS.num_demos):

        # Log custom metadata during new episode: language embeddings randomly.
        if FLAGS.enable_envlogger:
            env.set_episode_metadata({
                "language_embedding": np.random.random((5,)).astype(np.float32)
            })
            env.set_step_metadata({"timestamp": time.time()})

        logging.info('episode %r', i)
        
        # Start a new episode
        env.reset()
        terminated = False
        truncated = False

        step = 0

        # Termination occurs when the hand reaches the target or the maximum trajectory length is reached
        while not (terminated or truncated):
            
            # Get action from the demo function
            action = get_kb_demo_action()

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

            print(f" step: {step}", f"reward: {reward:.3f}\n")
            step += 1

    logging.info(
        'Done training a random agent for %r episodes.', FLAGS.num_demos)
    
    #env.close() # it seems async_drq_sim does not use close() dm_env method error.

if __name__ == '__main__':
    app.run(main)