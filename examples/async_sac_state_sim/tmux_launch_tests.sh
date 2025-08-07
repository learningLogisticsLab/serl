#!/bin/bash


# Create a new tmux session
tmux new-session -d -s serl_session
tmux setw -g remain-on-exit on

for SEED in {1..5}
do
    # New window
    tmux new-window -t serl_session -n $SEED
    # Split the window horizontally
    tmux split-window -v
    tmux split-pane -h -t serl_session:$SEED.1

    # Find open ports for TrainerServer
    OPEN_PORTS=$( comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 2 )
    PORTS=( $OPEN_PORTS )
    # Navigate to the activate the conda environment in the first pane
    tmux send-keys -t serl_session:$SEED.0 "bash automated_tests.sh $SEED ${PORTS[0]} ${PORTS[1]}" C-m
    echo ${PORTS[0]}
    echo ${PORTS[1]}

done
# Attach to the tmux session
tmux attach-session -t serl_session

# kill the tmux session by running the following command
# tmux kill-session -t serl_session
