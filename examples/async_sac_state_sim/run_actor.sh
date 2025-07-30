#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \

python async_sac_state_sim.py \
    --actor \
    --env PandaReachCube-v0 \
    --exp_name PandaReachCube-v0_state_sim_3D_con-81-3-batch_2048_replay_8M_utd_32 \
    --seed 0 \
    --replay_buffer_type fractal_symmetry_replay_buffer \
    --max_steps 300_000 \
    --training_starts 1000 \
    --critic_actor_ratio 32 \
    --batch_size 2048 \
    --replay_buffer_capacity 8_000_000 \
    --save_model True \
    --branch_method contraction \
    --split_method time \
    --max_traj_length 100 \
    --max_depth 4 \
    --start_num 81 \
    --alpha 1 \
    --max_depth 4 \
    --branching_factor 3 \
    --workspace_width 0.5 \
    --render
    # --checkpoint_period 10000 \
    # --checkpoint_path "$CHECKPOINT_DIR" \
    #--render \
    #--debug # wandb is disabled when debug
