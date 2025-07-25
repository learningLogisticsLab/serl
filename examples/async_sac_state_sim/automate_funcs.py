import subprocess
import os
import psutil
from absl import app
import copy

LOG_DIR = "serl_logs"
SESSION = "serl_session"

os.makedirs(LOG_DIR, exist_ok=True)

def run_tmux_command(cmd: str):
    return subprocess.run(cmd, shell=True, check=True)

def set_conda_env(session_name: str,
                  env: str = "serl", 
                  workdir: str = "./"):
    """
    Set the conda environment for the current shell in a tmux session.
    Call as `set_conda_env(f"{SESSION}:0.0", workdir="./", env="serl")`
    """

    cmd = (
        f"conda activate {env} && "
        f"cd {workdir}"
    )

    subprocess.run(
        f'tmux send-keys -t {session_name} "{cmd}" C-m',
        shell=True
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL
    )    

def send_tmux_command(target: str, command: str):
    """
    Send a command to a specific tmux target session as: 
    (f"{SESSION}:0.0", f"bash ./automate_actor.sh")
    Args:
     - target: str - The tmux target session (e.g., "serl_session:0.0")
     - command: str - The command to run in the tmux session

    Returns:
     - None

    """
    full_command = (
        f"{command}"
    )
    
    subprocess.run(
        f'tmux send-keys -t {target} "{full_command}" C-m',
        shell=True,
        #check=True
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
        # Session does not exist, create it and split the window
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "-n", "main"], check=True)
        subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0"], check=True)

        # Set the conda environment in both panes once at the beginning
        set_conda_env(f"{SESSION}:0.0", workdir="./", env="serl")
        set_conda_env(f"{SESSION}:0.1", workdir="./", env="serl")

def getProcesses(name):
    p1_pid = None
    p2_pid = None
    first = True
    while True:
        for proc in psutil.process_iter(["name"]):
            if proc.info["name"] == name:
                if first:
                    p1_pid = proc.pid
                    first = False
                else:
                    p2_pid = proc.pid
                    break
        if p1_pid and p2_pid and p1_pid != p2_pid:
            p1 = psutil.Process(p1_pid)
            p2 = psutil.Process(p2_pid)
            return p1, p2
        
def run_test(args:dict):

    max_steps = args["max_steps"]
    for seed in range(2):
        
        args["seed"] = seed
        actor_log = os.path.join(LOG_DIR, f"actor_seed_{seed}.log")
        learner_log = os.path.join(LOG_DIR, f"learner_seed_{seed}.log")

        args["max_steps"] = max_steps * 10000
        send_tmux_command(f"{SESSION}:0.0", f"bash ./automate_actor.sh {' '.join(f'--{k} {v}' for k, v in args.items())}") #conda/jax/tmux mem conflict? , f"{actor_cmd(args)}")# > {actor_log} 2>&1") # Uncomment to send to log files
        args["max_steps"] = max_steps
        send_tmux_command(f"{SESSION}:0.1", f"bash ./automate_learner.sh {' '.join(f'--{k} {v}' for k, v in args.items())}")#mem conflict?, f"{learner_cmd(args)}")# > {learner_log} 2>&1")

        p1, p2 = getProcesses("async_sac_state")

        while (True):
            if p1.is_running():
                if p2.is_running():
                    continue
                else:
                    p1.terminate()
                    p1.wait()
                    break
            else:
                if p2.is_running():
                    p2.terminate()
                    p2.wait()
                    break
            break

           

def main(_):
    ensure_tmux_session(SESSION)
    set_conda_env

    env = "PandaReachCube-v0"

    base_args = {
        "env": "PandaReachCube-v0",
        "random_steps": 1000,
        "training_starts": 1000,
        "critic_actor_ratio": 8,
        "batch_size": 256,
        "replay_buffer_capacity": 1000000,
        "save_model": True,
        "max_steps": 50000,
        "workspace_width": 0.5
    }

    # --- Baseline ---
    args = copy.deepcopy(base_args)
    args["replay_buffer_type"] = "replay_buffer"
    args["exp_name"] = f"{args['env']}-baseline"
    run_test(args)     
        

    # --- 1x1, 3x3, 9x9, 27x27 ---
    args = copy.deepcopy(base_args)
    args["replay_buffer_type"] = "fractal_symmetry_replay_buffer"
    args["branch_method"] = "constant"
    args["split_method"] = "constant"

    for b in (1, 3, 9, 27):
        args["starting_branch_count"] = b
        args["exp_name"] = f"{args['env']}-{b}x{b}"
        run_test(args) 

    # --- Fractal Expansion 3^4, 9^2 ---
    args = copy.deepcopy(base_args)
    args["replay_buffer_type"] = "fractal_symmetry_replay_buffer"
    args["branch_method"] = "fractal"
    args["split_method"] = "time"
    args["alpha"] = 1

    for b, d in ((3, 2), (3, 3)):
        for a in (0.25, 0.5, 0.75, 1):
            args["branching_factor"] = b
            args["max_depth"] = d
            args["exp_name"] = f"{args['env']}-{b}^{d}-alpha-{a}"
            run_test(args) 

if __name__ == "__main__":
    app.run(main)
