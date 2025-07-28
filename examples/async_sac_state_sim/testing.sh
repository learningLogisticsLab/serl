#!/bin/bash

# Tmux options
tmux set-option -g history-limit 100000

# Exit on error: prevents silent failures and makes bugs easier to catch:
# -e: exit immediately on command failure
# -u: error on use of undefined variables
# -o pipefail: error if any command in a pipeline fails
set -euo pipefail  

# Safe IFS for word splitting. Protects against word-splitting bugs due to unexpected whitespace in filenames or variables.
IFS=$'\n\t'         

# === CONFIGURATION ===
env="PandaReachCube-v0"
session="test"
window="0"

# === CLEANUP ON EXIT ===
# Kill only background jobs started by the script
trap 'jobs -p | xargs -r kill' EXIT INT TERM

# === START TMUX SESSION ===
if ! tmux has-session -t "$session" 2>/dev/null; then # suppress error output if session does not exist
    tmux new-session -d -s "$session" -n main         # start new detached tmux session named $session with init window named main
else
    echo "Session $session already exists. Reusing..."
fi

# === HELPER FUNCTION ===
# Centralizes the logic for sending commands into tmux. Improves readability and makes it easy to add logging or future logic.
send_tmux_command() {
    local cmd="$1"                                  # Take 1st arg in func & store in var cmd.
    tmux send-keys -t "$session:$window" "$cmd" C-m # send command to specific tmux window
}

# === BASELINE RUNS ===
# tmux_script_launch.sh must have ./ in front to tell tmux to execute the script in the current directory. 
for seed in {1}; do # Loop 
    send_tmux_command "./tmux_script_launch.sh --env $env --name ${env}-baseline --type replay_buffer --seed $seed"
done
