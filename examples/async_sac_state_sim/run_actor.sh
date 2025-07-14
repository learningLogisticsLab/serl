export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export ENV_NAME="PandaReachCube-v0" && \
python async_sac_state_sim.py "$@" \
    --actor \
    --render \
    --env $ENV_NAME \
    --exp_name=serl-reach \
    --seed 0 \
    --random_steps 1000 \
    --max_steps 1000000 \
    --training_starts 1000 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --save_model True \
    --checkpoint_period 10000 \
    --checkpoint_path "$SCRIPT_DIR/$ENV_NAME/checkpoints" \ 
    #--debug # wandb is disabled when debug

