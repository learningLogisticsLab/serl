
import subprocess
from absl import app
import time

def actor(kwargs:dict):
    script = "python3 /home/ryan_vanderstelt/serl/examples/async_sac_state_sim/async_sac_state_sim.py --actor"
    for k, v in kwargs.items():
        script += f" --{k}"
        script += f" {v}"
    return script

def learner(kwargs:dict):
    script = "python3 /home/ryan_vanderstelt/serl/examples/async_sac_state_sim/async_sac_state_sim.py --learner"
    for k, v in kwargs.items():
        script += f" --{k}"
        script += f" {v}"
    return script

def main(_):
    env = "PandaReachCube-v0"
    random_steps = 1000
    max_steps_learner = 10000
    max_steps_actor = max_steps_learner * 2
    training_starts = 1000
    critic_actor_ratio = 8
    batch_size = 256
    replay_buffer_capacity = 1000000
    save_model = True

    rb_args = {
            "env": env,
            "random_steps": random_steps,
            "max_steps": max_steps_actor,
            "training_starts": training_starts,
            "critic_actor_ratio": critic_actor_ratio,
            "batch_size": batch_size,
            "replay_buffer_capacity": replay_buffer_capacity,
            "save_model": save_model,
        }

    subprocess.Popen(['tmux', 'new-session', '-d', '-s', 'serl_session'], shell=True)
    time.sleep(10)
    subprocess.run("tmux split-window -v", shell=True)

    # baseline
    rb_args["replay_buffer_type"] = "replay_buffer"
    rb_args["exp_name"] = f"{env}-baseline"
    for seed in range(0, 5):
        rb_args["seed"] = seed
        rb_args["max_steps"] = max_steps_actor
        actor_process = subprocess.Popen(f"tmux send-keys -t serl_session:0.0 \"conda activate serl && {actor(kwargs=rb_args)}\" C-m", shell=True)


        rb_args["max_steps"] = max_steps_learner
        learner_process = subprocess.Popen(f"tmux send-keys -t serl_session:0.1 \"conda activate serl && {learner(kwargs=rb_args)}\" C-m", shell=True)
        subprocess.run("tmux attach-session -t serl_session", shell=True)

        learner_process.wait()
        actor_process.terminate()
        actor_process.wait()

    # constant
    # rb_args["replay_buffer_type"] = "fractal_symmetry_replay_buffer"
    # rb_args["branch_method"] = "constant"
    # rb_args["split_method"] = "constant"
    # for b in [1, 3, 9, 27]:
    #     rb_args["starting_branch_count"] = b
    #     rb_args["exp_name"] = f"{env}-{b}x{b}"
    #     for seed in range(0, 5):
    #         rb_args["seed"] = seed
    #         rb_args["max_steps"] = max_steps_actor
    #         actor_process = subprocess.Popen(["tmux", "send-keys", "-t", "serl_session:0.0", actor(kwargs=rb_args)])

    #         rb_args["max_steps"] = max_steps_learner
    #         learner_process = subprocess.Popen(learner(kwargs=rb_args), shell=True)

    #         learner_process.wait()
    #         actor_process.terminate()
    #         actor_process.wait()


if __name__ == "__main__":
    app.run(main)
    