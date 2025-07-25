#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \

python async_sac_state_sim.py "$@"\
    --learner \
    --env PandaReachCube-v0 \
    --exp_name reach-baseline \
    --replay_buffer_type replay_buffer \
    --max_steps 80000 \
    --training_starts 1000 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --replay_buffer_capacity 100000 \
    --save_model True \
    # --branch_method fractal \
    # --split_method time \
    # --starting_branch_count 1 \
    # --alpha 1 \
    # --max_depth 4 \
    # --branching_factor 3 \
    # --workspace_width 5 \
    # --checkpoint_period 10000 \
    # --checkpoint_path "$CHECKPOINT_DIR" \
    #--debug # wandb is disabled when debug
