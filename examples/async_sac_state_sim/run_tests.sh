#!/bin/bash

env="PandaReachCube-v0"

trap "kill 0" EXIT INT TERM

# Baseline

cmd=""
for i in $(seq 1 5); do
    cmd+="bash -c 'tmux_launch.sh --env $env --name \'$env-baseline\' --type \'replay_buffer\' --seed $i'; "
done



branch=("constant" "linear" "fractal" "OIO" "IOI" "inv_linear" "inv_fractal")
for b in "${branch[@]}"; do
    # constant branching
    if [ $b == "constant" ]; then
        branch_factor=(1 3 9 27)
        for bf in "${branch_factor[@]}"; do
            b_num=$(( $bf*$bf ))
            for seed in $(seq 1 5); do
                cmd+="bash -c 'tmux_launch.sh --env $env --name \'$env-$branch-$b_num\' --type \'fractal_symmetry_replay_buffer\' --branch $b --starting_branch $bf --seed $seed'; "
            done
        done
    fi
done
tmux new-session -d -s testing_session
tmux send-keys -t testing_session:0.0 $cmd C-m
