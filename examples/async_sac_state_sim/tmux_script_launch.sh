#!/bin/bash

EXAMPLE_DIR=${EXAMPLE_DIR:-"async_sac_state_sim"}
CONDA_ENV=${CONDA_ENV:-"serl"}

echo "Running from $(pwd)"

# This script assumes it is being run inside an existing tmux session from testing.sh
# Split the current window vertically
tmux split-window -v

# Run actor in the top pane
tmux send-keys -t "$(tmux display-message -p '#S')":0.0 "conda activate $CONDA_ENV && bash run_actor.sh" C-m

# Run learner in the bottom pane
tmux send-keys -t "$(tmux display-message -p '#S')":0.1 "conda activate $CONDA_ENV && bash run_learner.sh" C-m

# Optionally, focus the upper pane again
tmux select-pane -t 0
