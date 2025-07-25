#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \

python async_sac_state_sim.py "$@" \
    --actor \
    --env PandaPickCube-v0 \
    --exp_name=serl_reach \
    --seed 0 \
    --random_steps 1000 \
    --load_demos \
    --demo_dir "/data/data/serl/demos" \
    --file_name "data_franka_reach_random_20.npz"
    #--debug # wandb is disabled when debug
    #--render 
