#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \

python async_sac_state_sim.py "$@"\
    --learner \
    --env PandaReachCube-v0 \
    --exp_name PandaReachCube-v0_async_sac_state_demos_5_replay_1M_batch_2048_utd_32 \
    --replay_buffer_type replay_buffer \
    --max_steps 50_000 \
    --training_starts 1000 \
    --random_steps 1000 \
    --critic_actor_ratio 32 \
    --batch_size 2048 \
    --replay_buffer_capacity 1_000_000 \
    --save_model True \
    --load_demos \
    --demo_dir /data/data/serl/demos \
    --file_name data_franka_reach_random_5_2.npz \    
    --branch_method contraction \
    --split_method time \
    --starting_branch_count 1 \
    --max_traj_length 100 \
    --start_num 81 \
    --alpha 1 \
    --max_depth 4 \
    --branching_factor 3 \
    --workspace_width 0.5 \
    # --checkpoint_period 10000 \
    # --checkpoint_path "$CHECKPOINT_DIR" \
    #--debug # wandb is disabled when debug
    #--render 
