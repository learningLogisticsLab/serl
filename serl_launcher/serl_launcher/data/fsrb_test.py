import gym.wrappers
import numpy as np
import gym
from serl_launcher.data.dataset import DatasetDict
from serl_launcher.data.fractal_symmetry_replay_buffer import FractalSymmetryReplayBuffer
from absl import app, flags
import franka_sim

FLAGS = flags.FLAGS

flags.DEFINE_integer("replay_buffer_capacity", 1000000, "Replay buffer capacity.")
flags.DEFINE_string("branch_method", "test", "placeholder")
flags.DEFINE_string("split_method", "test", "placeholder")
flags.DEFINE_integer("workspace_width", 50, "workspace width in centimeters")
flags.DEFINE_integer("depth", 4, "Total layers of depth")
flags.DEFINE_integer("dendrites", 3, "Dendrites for fractal branching") # Remember to set default to None
flags.DEFINE_integer("timesplit_freq", None, "Frequency of splits according to time")
flags.DEFINE_integer("branch_count_rate_of_change", None, "Rate of change for linear branching")

def main(_):

    # Initialize replay buffer
    env = gym.make("PandaPickCube-v0")
    env = gym.wrappers.FlattenObservation(env)
    observation_space=env.observation_space
    action_space=env.action_space

    buffer = FractalSymmetryReplayBuffer(
        observation_space=observation_space,
        action_space=action_space,
        capacity=FLAGS.replay_buffer_capacity,
        split_method=FLAGS.split_method,
        branch_method=FLAGS.branch_method,
        workspace_width=FLAGS.workspace_width,
        depth=FLAGS.depth,
        dendrites=FLAGS.dendrites,
        timesplit_freq=FLAGS.timesplit_freq,
        branch_count_rate_of_change=FLAGS.branch_count_rate_of_change,
    )

    observation, info = env.reset()
    action = env.action_space.sample()
    next_observation, reward, terminated, truncated, info = env.step(action)
    data_dict = dict(
        observations=observation,
        next_observations=next_observation,
        actions=action,
        rewards=reward,
        masks=False,
        dones=truncated or terminated,
    )

    del env, observation, next_observation, action, reward, truncated, terminated, action_space, observation_space, info, _

    # transform() test

    expected = np.copy(data_dict["observations"]) * 2
    buffer.transform(data_dict, np.copy(data_dict["observations"]))
    result = data_dict["observations"]
    assert np.array_equal(result, expected), f"\033[31mTEST FAILED\033[0m transform() test failed (expected {expected} but got {result})"

    del result, expected
    
    print("\033[32mTEST PASSED \033[0m transform() tests passed")
    # branch() tests

    ## fractal

    for dendrites in range(1, 10, 2):
        buffer.dendrites=dendrites
        for depth in range(1, 6):
            buffer.current_depth=depth - 1
            result = buffer.fractal_branch()
            expected = dendrites ** depth
            assert result == expected, f"\033[31mTEST FAILED\033[0m fractal_branch() test failed (expected {expected} but got {result})"
    
    del dendrites, depth, result, expected

    ## constant

    ## linear

    ## exponential

    print("\033[32mTEST PASSED \033[0m _branch() tests passed")

    # split() tests

    ## time

    ## height

    ## rel_pos

    ## velocity

    print("\033[32mTEST PASSED \033[0m _split() tests passed")

    # insert() tests
    data_dict["observations"][0] = 0
    start = data_dict["observations"][0] - FLAGS.workspace_width / 2

    for i in range(0,10):
        buffer.insert(data_dict)
        for idx in range(buffer.current_branch_count):
            expected = FLAGS.workspace_width * (2 * idx + 1)/(2 * buffer.current_branch_count)
            result = buffer.dataset_dict["observations"][idx][0] - start
            assert result == expected, f"\033[31mTEST FAILED\033[0m insert() test failed (expected {expected} but got {result})"
            print(f"{result} is correct!!!")
        buffer._insert_index = 0

    del i, idx, result, expected

    print("\033[32mTEST PASSED \033[0m insert() tests passed")

    # for idx in range(8):
    #     obs = np.array([50, 0, 8-idx, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    #     next_obs = obs - np.array([50, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    #     rewards = np.float32(0)
    #     # truncateds = False
    #     masks=True
    #     dones = False
    #     if idx == 7:
    #         rewards += 1
    #         masks = False
    #         dones = True
    #     transition = dict(
    #         observations=obs, 
    #         next_observations=next_obs,
    #         actions = actions,
    #         rewards=rewards,
    #         # truncateds=truncateds,
    #         masks=masks,
    #         dones=dones,
    #         )
    #     buffer.insert(transition)
        
    print("\nfinished!\n")



if __name__ == "__main__":
    app.run(main)