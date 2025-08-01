#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \

export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="/data/fsrb_testing/checkpoints-$TIMESTAMP" && \

# Create checkpoint directory if it doesn't exist
# if [ ! -d "$CHECKPOINT_DIR" ]; then
#     echo "Creating checkpoint directory: $CHECKPOINT_DIR"
#     mkdir -p "$CHECKPOINT_DIR" || {
#         echo "Failed to create checkpoint directory!" >&2
#         exit 1
#     }
# fi

python async_sac_state_sim.py "$@"\
    --learner \
    --env PandaReachCube-v0 \
    --exp_name PandaReachCube-v0_state_sim_3D_con-81-9-batch_256_replay_1M_utd_8_ww_const \
    --replay_buffer_type fractal_symmetry_replay_buffer \
    --seed 0 \
    --max_steps 30_000 \
    --training_starts 1000 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --replay_buffer_capacity 1_000_000 \
    --save_model True \
    --branch_method contraction \
    --split_method time \
    --max_traj_length 100 \
    --alpha 1 \
    --max_depth 2 \
    --branching_factor 3 \
    --workspace_width 0.5 \
    --start_num 81 \
    --workspace_width_method constant \
    #--debug \
    #--render \
    #--checkpoint_period 10000 \
    #--checkpoint_path "$CHECKPOINT_DIR" \
