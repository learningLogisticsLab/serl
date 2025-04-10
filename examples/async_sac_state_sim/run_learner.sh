export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_state_sim.py "$@" \
    --learner \
    --env PandaPickCube-v0 \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --max_steps 1000000 \
    --training_starts 1000 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --save_model True \
    --checkpoint_period 10000 \
    --checkpoint_path "/home/student/robot/robot_ws/src/serl/examples/async_sac_state_sim/checkpoints" \ 
    #--debug # wandb is disabled when debug
