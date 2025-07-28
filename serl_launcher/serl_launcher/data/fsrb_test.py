import gym.wrappers
import numpy as np
import gym
from serl_launcher.utils.launcher import make_replay_buffer
from absl import app
import franka_sim
from datetime import datetime
# import pandas as pd

def main(_):

    # Initialize replay buffer
    env = gym.make("PandaReachCube-v0")
    env = gym.wrappers.FlattenObservation(env)

    replay_buffer = make_replay_buffer(
        env,
        type="fractal_symmetry_replay_buffer",
        capacity=1000000,
        split_method="time",
        branch_method="fractal",
        workspace_width=0.5,
        x_obs_idx=np.array([0, 4]),
        y_obs_idx= np.array([1, 5]),
        alpha=1,
        max_traj_length=20,
        max_depth=4,
        branching_factor=3,
        starting_branch_count=1,
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

    del env, observation, next_observation, action, reward, truncated, terminated, info, _

    # temp testing

    # replay_buffer.branch =replay_buffer.constant_branch
    # replay_buffer.split =replay_buffer.constant_split

    # for b in (1, 3, 9, 27):
    #     avg = 0
    #     high = 0
    #     low = 10
    #     replay_buffer.starting_branch_count = b
    #     for _ in range(50):
    #         start=datetime.now()
    #         replay_buffer.insert(data_dict)
    #         duration=(datetime.now() - start).total_seconds()
    #         avg += duration
    #         if duration < low:
    #             low = duration
    #         if duration > high:
    #             high = duration
        
    #     avg = avg/50
    #     print(f"Test: {b}x{b}\n\nhigh: {high:.5f}\nlow: {low:.5f}\naverage: {avg:.5f}\n\n")

    # replay_buffer.branch =replay_buffer.fractal_branch
    # replay_buffer.split =replay_buffer.time_split

    # avg = 0
    # high = 0
    # low = 10
    # for b, d in ((3,4),(9,2)):
    #     for a in (0.25, 0.5, 0.75, 1):
    #         avg = 0
    #         high = 0
    #         low = 10
    #         replay_buffer.branching_factor=b
    #         replay_buffer.max_depth=d
    #         replay_buffer.current_depth=0
    #         for i in range(20):
    #             start=datetime.now()
    #             replay_buffer.insert(data_dict)
    #             duration=(datetime.now() - start).total_seconds()
    #             avg += duration
    #             if duration < low:
    #                 low = duration
    #             if duration > high:
    #                 high = duration

    #         avg = avg/50
    #         print(f"Test: {b}^{d} at alpha={a}\n\nhigh: {high:.5f}\nlow: {low:.5f}\naverage: {avg:.5f}\n\n")
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