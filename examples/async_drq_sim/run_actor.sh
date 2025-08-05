export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python async_drq_sim.py "$@" \
    --actor \
    --env PandaReachCube-v0 \
    --render \
    --exp_name PandaReachCube-v0-drq-baselines \
    --seed 0 \
    --random_steps 1000 \
    --max_steps 50_000 \
    --encoder_type resnet-pretrained \
    # --debug
