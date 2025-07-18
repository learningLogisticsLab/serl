import gym.wrappers
import numpy as np
import gym
import jax
from serl_launcher.utils.launcher import (
    make_replay_buffer,
    make_trainer_config
)
from agentlace.trainer import TrainerServer
from serl_launcher.data.replay_buffer import ReplayBuffer
from absl import app, flags
import franka_sim

def main(_):
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    env = gym.make("PandaReachCube-v0")
    env = gym.wrappers.FlattenObservation(env)

    replay_buffer = make_replay_buffer(
        env=env,
        observation_space=env.observation_space,
        action_space=env.action_space,
        capacity=100000,
    )

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": 256,
        },
        device=sharding.replicate(),
    )

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
    
        return {}  # not expecting a response
    
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    server.register_data_store("actor_env", replay_buffer)

    size = len(replay_buffer)

    batch = next(replay_iterator)

