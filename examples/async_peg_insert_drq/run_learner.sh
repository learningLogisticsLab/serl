export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.6 && \
python async_drq_randomized.py "$@" \
    --learner \
    --env FrankaPegInsert-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_peg_insert_random_resnet_097 \
    --seed 0 \
    --max_steps 25000 \
    --random_steps 1000 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 128 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path peg_insert_20_demos_2025-04-14_15-36-51.pkl\
    --checkpoint_period 1000 \
    --checkpoint_path /home/student/robot/robot_ws/src/serl/examples/async_peg_insert_drq/checkpoints