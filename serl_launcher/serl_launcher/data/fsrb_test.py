import gym.wrappers
import numpy as np
import gym
from serl_launcher.utils.launcher import make_replay_buffer
from serl_launcher.data.fractal_symmetry_replay_buffer import FractalSymmetryReplayBuffer
from absl import app, flags
import franka_sim
# import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_integer("capacity", 10000000, "Replay buffer capacity.")
flags.DEFINE_string("branch_method", "test", "Method for determining the number of transforms per dimension (x,y)")
flags.DEFINE_string("split_method", "test", "Method for determining whether to change the number of transforms per dimension (x,y)")
flags.DEFINE_float("workspace_width", 0.5, "workspace width in meters")
flags.DEFINE_integer("max_depth", 4, "Maximum level of depth") # For fractal_branch only
flags.DEFINE_integer("branching_factor", 3, "Rate of change of number of transforms per dimension (x,y)") # For fractal_branch only
flags.DEFINE_integer("starting_branch_count", 1, "Initial number of transforms per dimension (x,y)") # For constant_branch only

def main(_):

    x_obs_idx = np.array([0, 4])
    y_obs_idx = np.array([1, 5])

    # Initialize replay buffer
    env = gym.make("PandaReachCube-v0")
    env = gym.wrappers.FlattenObservation(env)

    replay_buffer = make_replay_buffer(
        env,
        type="fractal_symmetry_replay_buffer",
        capacity=FLAGS.capacity,
        split_method=FLAGS.split_method,
        branch_method=FLAGS.branch_method,
        workspace_width=FLAGS.workspace_width,
        x_obs_idx=x_obs_idx,
        y_obs_idx= y_obs_idx,
        max_depth=FLAGS.max_depth,
        branching_factor=FLAGS.branching_factor,
    )

    observation, info = env.reset()
    action = env.action_space.sample()
    next_observation, reward, terminated, truncated, info = env.step(action)
    data_dict = dict(
        observations=observation,
        next_observations=next_observation,
        actions=action,
        rewards=reward,
        masks=not truncated and not terminated,
        dones=truncated or terminated,
    )

    del env, observation, next_observation, action, reward, truncated, terminated, info, y_obs_idx, x_obs_idx, _

    # transform() test

    expected = np.copy(data_dict["observations"]) * 2
    replay_buffer.transform(data_dict, np.copy(data_dict["observations"]))
    result = data_dict["observations"]
    assert np.array_equal(result, expected), f"\033[31mTEST FAILED\033[0m transform() test failed (expected {expected} but got {result})"

    del result, expected
    
    print("\n\033[32mTEST PASSED \033[0m transform() test passed")

    # branch() tests

    ## fractal

    replay_buffer.branching_factor = 3

    replay_buffer.current_depth = 1
    result = replay_buffer.fractal_branch()
    expected = 3
    assert result == expected, f"\033[31mTEST FAILED\033[0m fractal_branch() test failed (expected {expected} but got {result})"
    
    replay_buffer.current_depth = 2
    result = replay_buffer.fractal_branch()
    expected = 9
    assert result == expected, f"\033[31mTEST FAILED\033[0m fractal_branch() test failed (expected {expected} but got {result})"
    
    replay_buffer.current_depth = 3
    result = replay_buffer.fractal_branch()
    expected = 27
    assert result == expected, f"\033[31mTEST FAILED\033[0m fractal_branch() test failed (expected {expected} but got {result})"
    
    replay_buffer.current_depth = 4
    result = replay_buffer.fractal_branch()
    expected = 81
    assert result == expected, f"\033[31mTEST FAILED\033[0m fractal_branch() test failed (expected {expected} but got {result})"
    
    replay_buffer.current_depth = 0

    del result, expected

    print("\033[32mTEST PASSED \033[0m fractal_branch() tests passed")

    # split() tests
    ## time
    replay_buffer.max_steps = 100
    replay_buffer.max_depth = 4

    replay_buffer.timestep = 0
    result = replay_buffer.time_split(data_dict)
    expected = True
    assert result == expected, f"\033[31mTEST FAILED\033[0m split() test failed (expected {expected} but got {result})"
    
    replay_buffer.timestep = 25
    result = replay_buffer.time_split(data_dict)
    expected = True
    assert result == expected, f"\033[31mTEST FAILED\033[0m split() test failed (expected {expected} but got {result})"

    replay_buffer.timestep = 50
    result = replay_buffer.time_split(data_dict)
    expected = True
    assert result == expected, f"\033[31mTEST FAILED\033[0m split() test failed (expected {expected} but got {result})"

    replay_buffer.timestep = 75
    result = replay_buffer.time_split(data_dict)
    expected = True
    assert result == expected, f"\033[31mTEST FAILED\033[0m split() test failed (expected {expected} but got {result})"


    replay_buffer.timestep = 100
    result = replay_buffer.time_split(data_dict)
    expected = False
    assert result == expected, f"\033[31mTEST FAILED\033[0m split() test failed (expected {expected} but got {result})"

    del result, expected

    print("\033[32mTEST PASSED \033[0m time_split() test passed")

    # insert() tests
    # insert() tests
    initial_size = len(replay_buffer.dataset_dict['observations'][0]) * replay_buffer._insert_index % len(replay_buffer.dataset_dict['observations'])

    replay_buffer.insert(data_dict)
    final_size = len(replay_buffer.dataset_dict['observations'][0]) * replay_buffer._insert_index % len(replay_buffer.dataset_dict['observations'])
    
    result = final_size > initial_size
    expected = True
    assert result == expected, f"\033[31mTEST FAILED\033[0m insert() test failed (expected buffer size to increase from {initial_size} to {final_size})"
    del result, expected, initial_size, final_size




    print("\033[32mTEST PASSED \033[0m insert() tests passed")




    print("\033[32mTEST PASSED \033[0m insert() tests passed")
        
    print("\nfinished!\n")



if __name__ == "__main__":
    app.run(main)