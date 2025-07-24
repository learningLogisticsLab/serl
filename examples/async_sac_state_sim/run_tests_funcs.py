import subprocess
import os
from absl import app

LOG_DIR = "serl_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def actor_cmd(kwargs: dict) -> str:
    cmd = "python async_sac_state_sim.py --actor"
    for k, v in kwargs.items():
        cmd += f" --{k} {v}"
    return cmd

def learner_cmd(kwargs: dict) -> str:
    cmd = "python async_sac_state_sim.py --learner"
    for k, v in kwargs.items():
        cmd += f" --{k} {v}"
    return cmd

def run_tmux_command(cmd: str):
    return subprocess.run(cmd, shell=True, check=True)

def send_tmux_command(target: str, command: str, env: str = "serl", workdir: str = "./"):
    
    full_command = (
        f"conda activate {env} && "
        f"cd {workdir} && "
        f"{command}"
    )
    
    subprocess.Popen(
        f'tmux send-keys -t {target} "{full_command}" C-m',
        shell=True
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL
    )

def ensure_tmux_session(session_name: str):
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name]
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL
    )
    if result.returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "-n", "main"], check=True)
        subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0"], check=True)

def main(_):
    session = "serl_session"
    ensure_tmux_session(session)

    env = "PandaReachCube-v0"
    random_steps = 1000
    max_steps_learner = 10000
    max_steps_actor = max_steps_learner * 2
    training_starts = 1000
    critic_actor_ratio = 8
    batch_size = 256
    replay_buffer_capacity = 1000000
    save_model = True

    base_args = {
        "env": env,
        "random_steps": random_steps,
        "training_starts": training_starts,
        "critic_actor_ratio": critic_actor_ratio,
        "batch_size": batch_size,
        "replay_buffer_capacity": replay_buffer_capacity,
        "save_model": save_model,
    }

    # --- Baseline ---
    print("Running baseline...")
    args = base_args.copy()
    args["replay_buffer_type"] = "replay_buffer"
    args["exp_name"] = f"{env}-baseline"

    for seed in range(1):
        args["seed"] = seed
        args["max_steps"] = max_steps_actor
        actor_log = os.path.join(LOG_DIR, f"actor_seed_{seed}.log")
        learner_log = os.path.join(LOG_DIR, f"learner_seed_{seed}.log")

        send_tmux_command(f"{session}:0.0", f"./run_actor.sh {' '.join(f'--{k} {v}' for k, v in args.items())}") #conda/jax/tmux mem conflict? , f"{actor_cmd(args)}")# > {actor_log} 2>&1") # Uncomment to send to log files
        args["max_steps"] = max_steps_learner
        send_tmux_command(f"{session}:0.1", f"./run_learner.sh {' '.join(f'--{k} {v}' for k, v in args.items())}")#mem conflict?, f"{learner_cmd(args)}")# > {learner_log} 2>&1")

        # Optional: Wait before next round
        print(f"Launched baseline with seed={seed}")

    # --- Fractal Symmetry Replay Buffer ---
    # print("Running fractal symmetry buffer variations...")
    # args = base_args.copy()
    # args["replay_buffer_type"] = "fractal_symmetry_replay_buffer"
    # args["branch_method"] = "constant"
    # args["split_method"] = "constant"

    # for b in [1, 3, 9, 27]:
    #     args["starting_branch_count"] = b
    #     args["exp_name"] = f"{env}-{b}x{b}"
    #     for seed in range(5):
    #         args["seed"] = seed
    #         args["max_steps"] = max_steps_actor
    #         actor_log = os.path.join(LOG_DIR, f"actor_b{b}_s{seed}.log")
    #         learner_log = os.path.join(LOG_DIR, f"learner_b{b}_s{seed}.log")

    #         send_tmux_command(f"{session}:0.0", f"{actor_cmd(args)} > {actor_log} 2>&1")
    #         args["max_steps"] = max_steps_learner
    #         send_tmux_command(f"{session}:0.1", f"{learner_cmd(args)} > {learner_log} 2>&1")

    #         print(f"Launched branch={b}, seed={seed}")

    # Optional: Attach manually to see progress
    # run_tmux_command(f"tmux attach-session -t {session}")

if __name__ == "__main__":
    app.run(main)
