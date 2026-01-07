# All export statements end with && \ to chain them together
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
# XLA memory fraction with learner+action <0.8. Learner needs more.
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
# Use malloc_async to reduce fragmentation, overlap memory allocation with compute, lower stalls and improve worklads. Requires cuda11.2+
export TF_GPU_ALLOCATOR=cuda_malloc_async && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export ENV_NAME="FrankaPegInsert-Vision-v0" && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/checkpoints-$TIMESTAMP" && \

# Create checkpoint directory if it doesn't exist
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Creating checkpoint directory: $CHECKPOINT_DIR"
    mkdir -p "$CHECKPOINT_DIR" || {
        echo "Failed to create checkpoint directory!" >&2
        exit 1
    }
fi

python async_drq_randomized.py "$@" \
    --learner \
    --render \
    --env $ENV_NAME \
    --exp_name="PegInsert-Random-Reset-N_DEMOS-TESTS_2x" \
    --random_steps 0 \
    --seed 4 \
    --training_starts 200 \
    --save_model \
    --batch_size 256 \
    --critic_actor_ratio 8 \
    --replay_buffer_capacity 800_000 \
    --random_steps 1_000 \
    --eval_period 2_000 \
    --encoder_type resnet-pretrained \
    --demo_path peg_insert_20_demos_2025-11-27_12-12-01.pkl \
    --save_model \
    --replay_buffer_type "memory_efficient_replay_buffer" \
    # --branch_method "constant" \
    # --starting_branch_count 27 \
    # --workspace_width 0.1 \
    # --eval_checkpoint_step=5000 \
    # --checkpoint_period 1000 \
    # --checkpoint_path "$CHECKPOINT_DIR" \
