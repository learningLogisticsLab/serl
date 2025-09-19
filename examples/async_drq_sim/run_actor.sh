export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
export MUJOCO_GL=egl && \

python async_drq_sim.py --actor "$@"
