#Run this file after any code changes to see if transforms are still being performed correctly

import gym.wrappers
import numpy as np
import gym
from transforms3d.euler import euler2mat
from serl_launcher.data.dataset import DatasetDict
from serl_launcher.data.ker_replay_buffer import KerReplayBuffer
from absl import app
import franka_sim

WORKSPACE_WIDTH =   0.5
CAPACITY = 1_000_000
MAX_Z_THETA = 0.1443
THETA_LIST = [np.pi/25, np.pi/30, np.pi/35, np.pi/40]
#Expected output for n_ker = 4 and the given theta_list
EXPECTED_OUTPUT = np.float32([
    [1.,1.,1.],[1.217273,-0.7198933,1.],
    [1.1860592,-0.7702359,1.],[1.1624864,-0.8053727,1.],
    [1.1441228,-0.8312539,1.],[1.,-1.,1.],
    [1.217273,0.7198933,1.],[1.1860592,0.7702359,1.],
    [1.1624864,0.8053727,1.],[1.1441228,0.8312539,1.]
                   ])
ACTUAL_OUTPUT = []

def tester(coord: np.array, thetas):
    transformed_coords = [coord]
    for theta in thetas:
        inv_theta = euler2mat(0, 0, -theta)
        theta = euler2mat(0, 0, theta)
        new_coord = np.dot(inv_theta, coord)
        new_coord[1] = -new_coord[1]
        new_coord = np.dot(theta, new_coord)
        transformed_coords.append(new_coord)

    for i in range(len(transformed_coords)):
        transformed_coords.append(np.array([transformed_coords[i][0], -transformed_coords[i][1], transformed_coords[i][2]]))

    return transformed_coords

def main(_):

    # Initialize replay buffer
    env = gym.make("PandaReachCube-v0")
    env = gym.wrappers.FlattenObservation(env)
    observation_space=env.observation_space
    action_space=env.action_space

    print(observation_space)
    print(action_space)

    buffer = KerReplayBuffer(
        observation_space=observation_space,
        action_space=action_space,
        capacity=CAPACITY,
        n_KER=4,
        max_z_theta=MAX_Z_THETA,
        z_theta_list=THETA_LIST
        )
    
    observation, info = env.reset()
    action = env.action_space.sample()
    next_observation, reward, terminated, truncated, info = env.step(action)

    generated_dict = dict(
        observations=np.float32([1, 1, 1, 0.1, 0.05, 0.0, 0.02, 0.45, 0.35, 0.15]),
        next_observations=np.float32([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.2, 0.0]),
        actions=np.float32(np.empty_like(action)),
        rewards=reward,
        masks=False,
        dones=truncated or terminated,
    )
    
    #print(f"\n\ndata_dict: \n{generated_dict}\n\n")

    transformed_data_dicts = buffer.ker_process(generated_dict)

    # tester_results = tester(generated_dict['observations'][0:3], thetas)

    for i in range(len(transformed_data_dicts)):
        if i == 0:
            print(f"\n\n\nOriginal transition: \n{transformed_data_dicts[i][0][0:3]}")
            ACTUAL_OUTPUT.append(transformed_data_dicts[i][0][0:3])
            # print(f"Original test transition: \n{tester_results[i]}\n\n")
            # if abs(tester_results[i][0] - transformed_data_dicts[i][0][0:3][0]) <= 0.0001 and abs(tester_results[i][1] - transformed_data_dicts[i][0][0:3][1]) <= 0.0001 and abs(tester_results[i][2] - transformed_data_dicts[i][0][0:3][2]) <= 0.0001:
            #     print('SAME')
        else:
            print(f"Transform {i}: \n\t{transformed_data_dicts[i][0][0:3]}\n")
            ACTUAL_OUTPUT.append(transformed_data_dicts[i][0][0:3])
            #\ttest: {tester_results[i]}
            # if abs(tester_results[i][0] - transformed_data_dicts[i][0][0:3][0]) <= 0.0001 and abs(tester_results[i][1] - transformed_data_dicts[i][0][0:3][1]) <= 0.0001 and abs(tester_results[i][2] - transformed_data_dicts[i][0][0:3][2]) <= 0.0001:
            #     print('SAME')
    
    # Is the expected output equal to the actual output?
    assert np.array_equal(EXPECTED_OUTPUT, ACTUAL_OUTPUT), f"ERROR: actual output:\n{ACTUAL_OUTPUT}\n\ndoes not match expected output:\n{EXPECTED_OUTPUT}"
    if np.array_equal(EXPECTED_OUTPUT, ACTUAL_OUTPUT):
        print("Output is as expected")
    buffer.insert(generated_dict, transformed_data_dicts)

if __name__ == "__main__":
    app.run(main)