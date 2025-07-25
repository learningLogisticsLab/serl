#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \

python async_sac_state_sim.py \
    --actor \
    --env PandaReachCube-v0 \
    --exp_name serl-reach \
    --seed 0 \
    --replay_buffer_type replay_buffer \
    --branch_method test \
    --split_method test \
    --starting_branch_count 1 \
    --max_steps 40000 \
    --training_starts 1000 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --replay_buffer_capacity 100000 \
    --save_model True \
    # --checkpoint_period 10000 \
    # --checkpoint_path "$CHECKPOINT_DIR" \
    #--render \
    #--debug # wandb is disabled when debug
