# Imports
import gym
import gym.wrappers
import gym.wrappers.flatten_observation
import numpy as np
from serl_launcher.data.dataset import DatasetDict
from serl_launcher.data.ker_replay_buffer import KerReplayBuffer
from absl import app, flags
import franka_sim


FLAGS = flags.FLAGS

flags.DEFINE_string('env', "PandaPickCube-v0", "gym environment id")
flags.DEFINE_float('workspace_width', 0.5, "width of robot workspace")
flags.DEFINE_integer('capacity', 1_000_000, "replay buffer capacity")
flags.DEFINE_float('max_z_theta', 0.1443, 'maximum angle used to generate reflectional symmetries')
flags.DEFINE_integer('n_KER', 8, 'Number of reflectional symmetries to create. Number of new transforms = (2*n_KER) - 1')

def main(_):

    # Create model and env  
    env = gym.make(FLAGS.env)
    env = gym.wrappers.FlattenObservation(env)
    observation_space = env.observation_space
    action_space = env.action_space

    # Create instance of KerReplayBuffer
    buffer = KerReplayBuffer(
        env=env,
        observation_space=observation_space,
        action_space=action_space,
        capacity=FLAGS.capacity,
        workspace_width=FLAGS.workspace_width,
        n_KER=FLAGS.n_KER,
        max_z_theta=FLAGS.max_z_theta
    )

    # Initialize environment
    observation, info = env.reset()
    action = env.action_space.sample()
    next_observation, reward, terminated, truncated, info = env.step(action)

    # Create datadict
    test_data_dict = dict(
        observations=np.float32([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
        next_observations=np.float32([0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]),
        actions=np.float32(np.empty_like(action)),
        rewards=reward,
        masks=False,
        dones=truncated or terminated,
    )

    # Insert into replay buffer with transformations
    transfomed_data_dicts = buffer.ker_process(test_data_dict)
    buffer.insert(test_data_dict, transfomed_data_dicts)


    # Sample from replay buffer to see if everything works correctly
    for i in range(2 * FLAGS.n_KER):
        print(f"\n\nData_dict at index {i}:\n")
        print(f'observations: {buffer.dataset_dict["observations"][i]}')
        print(f'actions: {buffer.dataset_dict["actions"][i]}')
        print(f'next_observations: {buffer.dataset_dict["next_observations"][i]}')
        print(f'rewards: {buffer.dataset_dict["rewards"][i]}')
        print(f'masks: {buffer.dataset_dict["masks"][i]}')
        print(f'dones: {buffer.dataset_dict["dones"][i]}')



if __name__ == "__main__":
    app.run(main)