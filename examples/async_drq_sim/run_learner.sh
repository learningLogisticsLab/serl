export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python async_drq_sim.py "$@" \
    --learner \
    --env PandaReachCube-v0 \
    --exp_name PandaReachCube-v0-drq-baselines \
    --seed 0 \
    --training_starts 1_000 \
    --max_steps 50_000 \
    --critic_actor_ratio 4 \
    --encoder_type resnet-pretrained \
    --demo_path franka_lift_cube_image_20_trajs.pkl \
    # --debug # wandb is disabled when debug
