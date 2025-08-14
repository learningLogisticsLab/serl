#!/bin/bash


# Create a new tmux session
tmux new-session -d -s serl_session
tmux setw -g remain-on-exit on

for SEED in 1
do
    # New window
    tmux new-window -t serl_session -n $SEED
    # Split the window horizontally
    tmux split-window -v
    tmux split-pane -h -t serl_session:$SEED.1

    # Navigate to the activate the conda environment in the first pane
    tmux send-keys -t serl_session:$SEED.0 "bash automated_tests.sh $SEED" C-m

done
# Attach to the tmux session
tmux attach-session -t serl_session

# kill the tmux session by running the following command
# tmux kill-session -t serl_session
