#!/usr/bin/env python3
import os
import time
from datetime import datetime

import numpy as np

# Logging
from absl import app, flags, logging
from oxe_envlogger.envlogger import AutoOXEEnvLogger

# DRL
import gym
import mujoco
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

# Needed to create the franka environment
import franka_sim

# Teleoperation imports
import sys, select, termios, tty

#-------------------------------------------------------------------------------------------
# Flags
#-------------------------------------------------------------------------------------------
flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Name of environment.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 200, "Maximum length of trajectory.")
flags.DEFINE_boolean("debug", True, "Debug mode.")  # debug mode will disable wandb logging
#flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")
flags.DEFINE_string("output_dir", "/data/data/serl/demos/franka_reach_drq_demo_script",
                    "Directory to save the output data. This is where the RLDS logs will be saved.")                     
flags.DEFINE_integer("num_demos", 2, "Number of episodes to log.")
flags.DEFINE_boolean("enable_envlogger", True, "Enable envlogger.")
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
def ensure_dir_exists():
    """
    Ensure that the output directory exists. If it does not exist, create it.

    Returns
    -------
    out_path : str
        The path to the output directory.
    """
    # Create a timestamped directory for saving outputs
    robot = "franka"
    task = "reach"
    initStateSpace = "random"  # "fixed" or "random"

    # Customize the path
    script_dir = FLAGS.output_dir   

    # Create output filename with configuration details
    fileName = "data_" + robot + "_" + task
    fileName += "_" + initStateSpace
    fileName += "_" + str(FLAGS.num_demos)
    
    # Add timestamp to filename for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fileName += "_" + timestamp

    # Build a filename in that same directory
    out_path = os.path.join(script_dir, fileName)    

    # Ensure the directory exists
    if not os.path.exists(out_path):
        os.makedirs(script_dir, exist_ok=True)
        logging.info(f"Created output directory at: {out_path}")
    else:
        logging.info(f"Output directory already exists at: {out_path}")

    return out_path

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
        viewer.cam.distance = 2.0            # Camera distance
        viewer.cam.azimuth = 155             # 0 = right, 90 = front, 180 = left
        viewer.cam.elevation = -30           # Negative = above, positive = below

        # Hide menu
        viewer._hide_overlay = True
    
    return viewer

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
        # env = SERLObsWrapper(env)
        env = SERLObsWrapper(
            env,
            target_hw=(128, 128),
            img_dtype=np.uint8,   # or np.float32
            normalize=False,      # True if using float32 in [0,1]
        )
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    logging.info(f'Done creating {FLAGS.env} environment.')

    if hasattr(env.unwrapped, '_viewer'):
        viewer = set_front_cam_view(env)
        if viewer:
            logging.info('Camera view set to front-facing perspective.')
        else:
            logging.warning('Failed to set camera view. Viewer not available.')

    # If envlogger is enabled, wrap the environment with AutoOXEEnvLogger to log episodes
    if FLAGS.enable_envlogger:

        out_path = ensure_dir_exists()

        env = AutoOXEEnvLogger(
            env=env,
            dataset_name=FLAGS.env,
            directory=out_path,
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