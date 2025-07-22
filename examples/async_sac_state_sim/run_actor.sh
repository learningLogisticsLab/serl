#!/bin/bash

# Default variable values
seed=0
env="PandaReachCube-v0"
branch="test"
type="fractal_symmetry_replay_buffer"
exp_name="untitled_experiment"
starting_branch_count=1

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

                exp_name=$(extract_argument $@)

                shift
                ;;
            --starting_branch*)
                if ! has_argument $@; then
                    echo "Starting_branch needs a positive odd integer" >&2
                    exit 1
                fi

                starting_branch_count=$(extract_argument $@)

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

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="/data/fsrb_testing/checkpoints-$TIMESTAMP" && \

# Create checkpoint directory if it doesn't exist
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Creating checkpoint directory: $CHECKPOINT_DIR"
    mkdir -p "$CHECKPOINT_DIR" || {
        echo "Failed to create checkpoint directory!" >&2
        exit 1
    }
fi

python async_sac_state_sim.py \
    --actor \
    --env $env \
    --exp_name=$exp_name \
    --seed $seed \
    --replay_buffer_type $type \
    --branch_method $branch \
    --starting_branch_count $starting_branch_count \
    --random_steps 1000 \
    --max_steps 100000 \
    --training_starts 1000 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --replay_buffer_capacity 1000000 \
    --save_model True \
    # --checkpoint_period 10000 \
    # --checkpoint_path "$CHECKPOINT_DIR" \
    #--render \
    #--debug # wandb is disabled when debug
