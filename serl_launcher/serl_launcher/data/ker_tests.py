import gym.wrappers
import numpy as np
import gym
from serl_launcher.data.dataset import DatasetDict
from serl_launcher.data.ker_replay_buffer import KerReplayBuffer
import franka_sim

WORKSPACE_WIDTH =   0.5
CAPACITY = 1_000_000
MAX_Z_THETA = 0.1443

def main():

    # Initialize replay buffer
    env = gym.make("PandaPickCube-v0")
    env = gym.wrappers.FlattenObservation(env)
    observation_space=env.observation_space
    action_space=env.action_space

    print(observation_space)
    print(action_space)

    buffer = KerReplayBuffer(
        observation_space=observation_space,
        action_space=action_space,
        capacity=CAPACITY,
        workspace_width=WORKSPACE_WIDTH,
        n_KER=8,
        max_z_theta=MAX_Z_THETA
        )
    
    observation, info = env.reset()
    action = env.action_space.sample()
    next_observation, reward, terminated, truncated, info = env.step(action)
   
    generated_dict = dict(
        observations=np.float32([1, 1, 1, 0.1, 0.05, 0.0, 0.02, 0.45, 0.35, 0.15]),
        next_observations=np.float32([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.2, 0.0]),
        actions=np.float32([0.1, 0.1, 0.1, 0.0]),
        rewards=reward,
        masks=False,
        dones=truncated or terminated,
    )
    
    print(f"\n\ndata_dict: \n{generated_dict}\n\n")

    transformed_data_dicts = buffer.ker_process(generated_dict)
    for i in range(len(transformed_data_dicts)):
        if i == 0:
            print(f"\n\n\nOriginal transition: \n{transformed_data_dicts[i]}\n\n")
        else:
            print(f"Transform {i}: \n{transformed_data_dicts[i]}\n\n")
    
    buffer.insert(transformed_data_dicts, generated_dict)

if __name__ == "__main__":
    main()