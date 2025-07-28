#!/bin/bash

tmux new-session -d -s serl_session "python run_tests.py"

tmux attach-session -t serl_session

# kill the tmux session by running the following command
# tmux kill-session -t serl_session
