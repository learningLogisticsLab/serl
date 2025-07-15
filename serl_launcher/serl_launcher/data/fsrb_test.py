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
flags.DEFINE_string("branch_method", "fractal", "placeholder")
flags.DEFINE_string("split_method", "time", "placeholder")
flags.DEFINE_float("workspace_width", 0.5, "workspace width in centimeters")
flags.DEFINE_integer("depth", 4, "Total layers of depth")
flags.DEFINE_integer("dendrites", 3, "Dendrites for fractal branching") # Remember to set default to None
flags.DEFINE_integer("timesplit_freq", None, "Frequency of splits according to time")
flags.DEFINE_integer("branch_count_rate_of_change", None, "Rate of change for linear branching")
flags.DEFINE_integer("starting_branch_count", 1, "Initial number of branches(sort of, TBD)")

def main(_):

    x_obs_idx = np.array([0, 7])
    y_obs_idx = np.array([1, 8])

    # Initialize replay buffer
    env = gym.make("PandaPickCube-v0")
    env = gym.wrappers.FlattenObservation(env)

    replay_buffer = make_replay_buffer(
        env,
        type="fractal_symmetry_replay_buffer",
        capacity=FLAGS.capacity,
        split_method=FLAGS.split_method,
        branch_method=FLAGS.branch_method,
        workspace_width=FLAGS.workspace_width,
        x_obs_idx=x_obs_idx,
        depth=FLAGS.depth,
        dendrites=FLAGS.dendrites,
        timesplit_freq=FLAGS.timesplit_freq,
        branch_count_rate_of_change=FLAGS.branch_count_rate_of_change,
        starting_branch_count=FLAGS.starting_branch_count,
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

    del env, observation, next_observation, action, reward, truncated, terminated, info, _

    # transform() test

    expected = np.copy(data_dict["observations"]) * 2
    replay_buffer.transform(data_dict, np.copy(data_dict["observations"]))
    result = data_dict["observations"]
    assert np.array_equal(result, expected), f"\033[31mTEST FAILED\033[0m transform() test failed (expected {expected} but got {result})"

    del result, expected
    
    print("\n\033[32mTEST PASSED \033[0m transform() tests passed")
    # branch() tests

    ## fractal

    # for dendrites in range(1, 10, 2):
    #     replay_buffer.dendrites=dendrites
    #     for depth in range(1, 6):
    #         replay_buffer.current_depth=depth - 1
    #         result = replay_buffer.fractal_branch()
    #         expected = dendrites ** depth
    #         assert result == expected, f"\033[31mTEST FAILED\033[0m fractal_branch() test failed (expected {expected} but got {result})"
    
    # del dendrites, depth, result, expected

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

    # important_indices = np.array([0, 7])

    # num_branches = 5000
    # workspace_options = 20

    # data = dict(
    #     workspace_width = np.empty(shape=(num_branches//2 * workspace_options), dtype=np.float32),
    #     num_branches = np.empty(shape=(num_branches//2 * workspace_options), dtype=int),
    #     density = np.empty(shape=(num_branches//2 * workspace_options), dtype=np.float32)
    # )

    # for w in range(0, workspace_options):
        # total_success = 0.0
        # for i in range(1,num_branches, 2):
        #     finished = False
        #     buffer.current_branch_count = i
        #     data_dict["observations"][important_indices] = 0
    #         start = data_dict["observations"][important_indices[0]] - FLAGS.workspace_width / 2
    #         buffer.insert(data_dict)
    #         for idx in range(buffer.current_branch_count):
    #             expected = round(FLAGS.workspace_width * (2 * idx + 1)/(2 * buffer.current_branch_count), 3)
    #             result = round(buffer.dataset_dict["observations"][buffer._insert_index - buffer.current_branch_count + idx][important_indices[0]] - start, 3)
    #             # assert result == expected, f"\033[31mTEST FAILED\033[0m insert() test failed (expected {expected} but got {result})"
    #             if result != expected:
    #                 # print(f"At ww = {FLAGS.workspace_width}, maximum transforms = {i - 2}")
    #                 # print(f"{i} failed at {idx}")
    #                 finished = True

    #                 break
    #         if not finished:
    #             print(f"SUCCESS at {i} branches")
    #             total_success += 1
    #         data["workspace_width"][i//2 + (num_branches//2) * w] = 0.05 * (w + 1)
    #         data["num_branches"][i//2 + (num_branches//2) * w] = i
    #         data["density"][i//2 + (num_branches//2) * w] = total_success/(i//2 + 1)

    #     FLAGS.workspace_width = FLAGS.workspace_width + 0.05
    #     buffer.workspace_width = FLAGS.workspace_width
    # del i, idx, result, expected

    # df = pd.DataFrame(data)
    # df.to_excel('output.xlsx', index=False)

    data_dict["observations"][0] = 0
    start = data_dict["observations"][0] - FLAGS.workspace_width / 2

    
    # buffer.insert(data_dict)
    # for idx in range(buffer.current_branch_count):
    #     expected = round(FLAGS.workspace_width * (2 * idx + 1)/(2 * buffer.current_branch_count), 3)
    #     result = round(buffer.dataset_dict["observations"][idx][0] - start, 3)
    #     assert result == expected, f"\033[31mTEST FAILED\033[0m insert() test failed (expected {expected} but got {result})"

    replay_buffer.insert(data_dict)
    for idx in range(replay_buffer.current_branch_count):
        expected = round(FLAGS.workspace_width * (2 * idx + 1)/(2 * replay_buffer.current_branch_count), 3)
        result = round(replay_buffer.dataset_dict["observations"][idx][0] - start, 3)
        assert result == expected, f"\033[31mTEST FAILED\033[0m insert() test failed (expected {expected} but got {result})"

    print("\033[32mTEST PASSED \033[0m insert() tests passed")
        
    print("\nfinished!\n")



if __name__ == "__main__":
    app.run(main)