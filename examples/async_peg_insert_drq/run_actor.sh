export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/checkpoints-$TIMESTAMP" && \
export CHECKPOINT_EVAL="/home/student/code/cleiver/serl/examples/async_peg_insert_drq/checkpoints/checkpoints-07-14-2025-23-15-59" && \


# Create checkpoint directory if it doesn't exist
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Creating checkpoint directory: $CHECKPOINT_DIR"
    mkdir -p "$CHECKPOINT_DIR" || {
        echo "Failed to create checkpoint directory!" >&2
        exit 1
    }
fi

python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env "FrankaPegInsert-Vision-v0" \
    --exp_name="PegInsert-baselines" \
    --max_steps 25000 \
    --seed 3 \
    --random_steps 1000 \
    --training_starts 200 \
    --encoder_type resnet-pretrained \
    --demo_path peg_insert_30_demos_2025-10-20_17-40-38.pkl \
    # --save_model \
    # --checkpoint_period 1000 \
    # --checkpoint_path "$CHECKPOINT_DIR" \
    # --eval_checkpoint_step=5000 \
    #--debug # wandb is disabled when debug
