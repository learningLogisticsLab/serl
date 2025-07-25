export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export ENV_NAME="PandaReachCube-v0" && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="$SCRIPT_DIR/$ENV_NAME_checkpoints/checkpoints-$TIMESTAMP" && \

# Create checkpoint directory if it doesn't exist
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Creating checkpoint directory: $CHECKPOINT_DIR"
    mkdir -p "$CHECKPOINT_DIR" || {
        echo "Failed to create checkpoint directory!" >&2
        exit 1
    }
fi

python async_sac_state_sim.py "$@" \
    --learner \
    --env $ENV_NAME \
    --exp_name=KER \
    --seed 0 \
    --max_steps 50_000 \
    --training_starts 1000 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --replay_buffer_capacity 1_000_000 \
    --save_model True \
    --checkpoint_period 10000 \
    --checkpoint_path "$CHECKPOINT_DIR" \
    --symmetry reflection \
    --n_KER 4
    #--debug # wandb is disabled when debug
