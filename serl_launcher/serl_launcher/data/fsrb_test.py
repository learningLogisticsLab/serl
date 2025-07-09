import gym.wrappers
import numpy as np
import gym
from serl_launcher.data.dataset import DatasetDict
from serl_launcher.data.fractal_symmetry_replay_buffer import FractalSymmetryReplayBuffer
from absl import app, flags
import franka_sim

FLAGS = flags.FLAGS

flags.DEFINE_integer("replay_buffer_capacity", 1000000, "Replay buffer capacity.")
flags.DEFINE_string("branch_method", "fractal", "placeholder")
flags.DEFINE_string("split_method", "test", "placeholder")
flags.DEFINE_integer("workspace_width", 50, "workspace width in centimeters")
flags.DEFINE_integer("depth", 4, "Total layers of depth")
flags.DEFINE_integer("dendrites", 3, "Dendrites for fractal branching") # Remember to set default to None
flags.DEFINE_integer("timesplit_freq", None, "Frequency of splits according to time")
flags.DEFINE_integer("branch_count_rate_of_change", None, "Rate of change for linear branching")

def main(_):

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

    obs = env.observation_space.sample()
    actions = env.action_space.sample()

    actions = np.array([1, 2, 3, 4], dtype=np.float32)

    for idx in range(8):
        obs = np.array([50, 0, 8-idx, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        next_obs = obs - np.array([50, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        rewards = np.float32(0)
        # truncateds = False
        masks=True
        dones = False
        if idx == 7:
            rewards += 1
            masks = False
            dones = True
        transition = dict(
            observations=obs, 
            next_observations=next_obs,
            actions = actions,
            rewards=rewards,
            # truncateds=truncateds,
            masks=masks,
            dones=dones,
            )
        buffer.insert(transition)
        
    print("finished\n")



if __name__ == "__main__":
    app.run(main)