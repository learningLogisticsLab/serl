#!/bin/bash

# Default variable values
seed=0
env="PandaReachCube-v0"
branch="test"
type="fractal_symmetry_replay_buffer"
name="untitled_experiment"
starting_branch=1

has_argument() {
    [[ ("$1" == *=* && -n ${1#*=}) || ( ! -z "$2" && "$2" != -*)  ]];
}

extract_argument() {
  echo "${2:-${1#*=}}"
}

# Function to handle options and arguments
handle_options() {
    while [ $# -gt 0 ]; do
        case $1 in
            -e | --env*)
                if ! has_argument $@; then
                    echo "Environment not specified" >&2
                    exit 1
                fi

                env=$(extract_argument $@)

                shift
                ;;
            -s | --seed*)
                if ! has_argument $@; then
                    echo "Seed needs a positive integer value" >&2
                    exit 1
                fi

                seed=$(extract_argument $@)

                shift
                ;;
            -t | --type*)
                if ! has_argument $@; then
                    echo "Type needs a string" >&2
                    exit 1
                fi

                type=$(extract_argument $@)

                shift
                ;;
            -n | --name*)
                if ! has_argument $@; then
                    echo "Name needs a string" >&2
                    exit 1
                fi

                name=$(extract_argument $@)

                shift
                ;;
            --starting_branch*)
                if ! has_argument $@; then
                    echo "Starting_branch needs a positive odd integer" >&2
                    exit 1
                fi

                starting_branch=$(extract_argument $@)

                shift
                ;;
            -b | --branch*)
                if ! has_argument $@; then
                    echo "Branch needs a string" >&2
                    exit 1
                fi

                branch=$(extract_argument $@)

                shift
                ;;
            *)
                echo "Invalid option: $1" >&2
                exit 1
                ;;
        esac
        shift
    done
}

# Main script execution
handle_options "$@"

# EXAMPLE_DIR=${EXAMPLE_DIR:-"examples/async_sac_state_sim"}
CONDA_ENV=${CONDA_ENV:-"serl"}

# cd $EXAMPLE_DIR
echo "Running from $(pwd)"

# Create a new tmux session
tmux new-session -d -s serl_session

# Split the window vertically
tmux split-window -v

# Navigate to the activate the conda environment in the first pane
tmux send-keys -t serl_session:0.0 "conda activate $CONDA_ENV && bash run_actor.sh --seed $seed --env $env --name $name --type $type --branch $branch --starting_branch $starting_branch" C-m

# Navigate to the activate the conda environment in the second pane
tmux send-keys -t serl_session:0.1 "conda activate $CONDA_ENV && bash run_learner.sh --seed $seed --env $env --name $name --type $type --branch $branch --starting_branch $starting_branch && tmux kill-session -t serl_session:0.0" C-m

# Attach to the tmux session
tmux attach-session -t serl_session

# kill the tmux session by running the following command
# tmux kill-session -t serl_session
