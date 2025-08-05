#!/bin/bash

tmux set-option -t serl_session:0.1 remain-on-exit on
tmux set-option -t serl_session:0.2 remain-on-exit on

TEST="async_sac_state_sim.py"
CONDA_ENV="serl"
ENV="PandaReachCube-v0"
MAX_STEPS=25000
TRAINING_STARTS=1000
RANDOM_STEPS=1000
CRITIC_ACTOR_RATIO=8
EXP_NAME="MASS-TESTING-$ENV-max-steps-$MAX_STEPS"

BASE_ARGS="--env $ENV --exp_name $EXP_NAME --training_starts $TRAINING_STARTS --random_steps $RANDOM_STEPS --critic_actor_ratio $CRITIC_ACTOR_RATIO"

echo "$BASE_ARGS"
for seed in 1 2 3
do
    for batch_size in 256
    do
        for replay_buffer_capacity in 1000000
        do
            # Baseline
            ARGS="--run_name baseline-batch-size-$batch_size-capacity-$replay_buffer_capacity --replay_buffer_type replay_buffer --batch_size $batch_size --replay_buffer_capacity $replay_buffer_capacity --seed $seed"

            tmux respawn-pane -k -t serl_session:0.1
            tmux respawn-pane -k -t serl_session:0.2
            tmux send-keys -t serl_session:0.1 "conda activate $CONDA_ENV && bash automated_tests_helper.sh --actor --max_steps $((MAX_STEPS * 2)) $BASE_ARGS $ARGS" C-m
            tmux send-keys -t serl_session:0.2 "conda activate $CONDA_ENV && bash automated_tests_helper.sh --learner --max_steps $MAX_STEPS $BASE_ARGS $ARGS" C-m "exit" C-m

            while ! tmux capture-pane -t serl_session:0.2 -p | grep "Pane is dead" > /dev/null; 
            do 
                sleep 1
                echo "I'm waiting :)"
            done
            echo "I'm DONE waiting >:("
                
            REPLAY_BUFFER_TYPE="fractal_symmetry_replay_buffer"
            for workspace_width in 0.5
            do
                for starting_branch_count in 1 3 9 27 81
                do 
                    # Constant
                    ARGS="--run_name constant-$starting_branch_count^1-workspace_width-$workspace_width-batch-size-$batch_size-capacity-$replay_buffer_capacity --replay_buffer_type $REPLAY_BUFFER_TYPE --batch_size $batch_size --replay_buffer_capacity $replay_buffer_capacity --seed $seed --workspace_width $workspace_width --branch_method 'constant' --starting_branch_count $starting_branch_count"

                    tmux respawn-pane -k -t serl_session:0.1
                    tmux respawn-pane -k -t serl_session:0.2
                    tmux send-keys -t serl_session:0.1 "conda activate $CONDA_ENV && bash automated_tests_helper.sh --actor --max_steps $((MAX_STEPS * 2)) $BASE_ARGS $ARGS" C-m
                    tmux send-keys -t serl_session:0.2 "conda activate $CONDA_ENV && bash automated_tests_helper.sh --learner --max_steps $MAX_STEPS $BASE_ARGS $ARGS" C-m "exit" C-m
                    while ! tmux capture-pane -t serl_session:0.2 -p | grep "Pane is dead" > /dev/null; 
                    do 
                        sleep 1
                        echo "I'm waiting :)"
                    done
                    echo "I'm DONE waiting >:("
                done

                for alpha in 0.9
                do
                    for branching_factor in 3 9
                    do
                        for max_depth in 2 4
                        do
                            # Fractal Expansion
                            ARGS="--run_name fractal_expansion-$branching_factor^$max_depth-alpha-$alpha-workspace_width-$workspace_width-batch-size-$batch_size-capacity-$replay_buffer_capacity --replay_buffer_type $REPLAY_BUFFER_TYPE --batch_size $batch_size --replay_buffer_capacity $replay_buffer_capacity --seed $seed --workspace_width $workspace_width --branch_method 'fractal' --alpha $alpha --branching_factor $branching_factor --max_depth $max_depth"
                            
                            tmux respawn-pane -k -t serl_session:0.1
                            tmux respawn-pane -k -t serl_session:0.2
                            tmux send-keys -t serl_session:0.1 "conda activate $CONDA_ENV && bash automated_tests_helper.sh --actor --max_steps $((MAX_STEPS * 2)) $BASE_ARGS $ARGS" C-m
                            tmux send-keys -t serl_session:0.2 "conda activate $CONDA_ENV && bash automated_tests_helper.sh --learner --max_steps $MAX_STEPS $BASE_ARGS $ARGS" C-m "exit" C-m
                            while ! tmux capture-pane -t serl_session:0.2 -p | grep "Pane is dead" > /dev/null; 
                            do 
                                sleep 1
                                echo "I'm waiting :)"
                            done
                            echo "I'm DONE waiting >:("

                            # Fractal Contraction
                            ARGS="--run_name fractal_contraction-$branching_factor^$max_depth-alpha-$alpha-workspace_width-$workspace_width-batch-size-$batch_size-capacity-$replay_buffer_capacity --replay_buffer_type $REPLAY_BUFFER_TYPE --batch_size $batch_size --replay_buffer_capacity $replay_buffer_capacity --seed $seed --workspace_width $workspace_width --branch_method 'contraction' --alpha $alpha --branching_factor $branching_factor --max_depth $max_depth"

                            tmux respawn-pane -k -t serl_session:0.1
                            tmux respawn-pane -k -t serl_session:0.2
                            tmux send-keys -t serl_session:0.1 "conda activate $CONDA_ENV && bash automated_tests_helper.sh --actor --max_steps $((MAX_STEPS * 2)) $BASE_ARGS $ARGS" C-m
                            tmux send-keys -t serl_session:0.2 "conda activate $CONDA_ENV && bash automated_tests_helper.sh --learner --max_steps $MAX_STEPS $BASE_ARGS $ARGS" C-m "exit" C-m
                            
                            while ! tmux capture-pane -t serl_session:0.2 -p | grep "Pane is dead" > /dev/null; 
                            do 
                                sleep 1
                                echo "I'm waiting :)"
                            done
                            echo "I'm DONE waiting >:("
                        done
                    done
                    for min_branch_count in 1 3 9
                    do
                        for max_branch_count in 3 9 27
                        do
                            if [ $min_branch_count -ge $max_branch_count ]; then
                                continue
                            fi
                            # Disassociative (Hourglass)
                            ARGS="--run_name disassociative-hourglass-$min_branch_count:$max_branch_count-alpha-$alpha-workspace_width-$workspace_width-batch-size-$batch_size-capacity-$replay_buffer_capacity --replay_buffer_type $REPLAY_BUFFER_TYPE --batch_size $batch_size --replay_buffer_capacity $replay_buffer_capacity --seed $seed --workspace_width $workspace_width --branch_method 'disassociated' --min_branch_count $min_branch_count --max_branch_count $max_branch_count --disassociated_type 'hourglass' --alpha $alpha"
                            
                            tmux respawn-pane -k -t serl_session:0.1
                            tmux respawn-pane -k -t serl_session:0.2
                            tmux send-keys -t serl_session:0.1 "conda activate $CONDA_ENV && bash automated_tests_helper.sh --actor --max_steps $((MAX_STEPS * 2)) $BASE_ARGS $ARGS" C-m
                            tmux send-keys -t serl_session:0.2 "conda activate $CONDA_ENV && bash automated_tests_helper.sh --learner --max_steps $MAX_STEPS $BASE_ARGS $ARGS" C-m "exit" C-m
                            
                            while ! tmux capture-pane -t serl_session:0.2 -p | grep "Pane is dead" > /dev/null; 
                            do 
                                sleep 1
                                echo "I'm waiting :)"
                            done
                            echo "I'm DONE waiting >:("

                            # Disassociative (Octahedron)
                            ARGS="--run_name disassociative-hourglass-$min_branch_count:$max_branch_count-alpha-$alpha-workspace_width-$workspace_width-batch-size-$batch_size-capacity-$replay_buffer_capacity --replay_buffer_type $REPLAY_BUFFER_TYPE --batch_size $batch_size --replay_buffer_capacity $replay_buffer_capacity --seed $seed --workspace_width $workspace_width --branch_method 'disassociated' --min_branch_count $min_branch_count --max_branch_count $max_branch_count --disassociated_type 'octahedron' --alpha $alpha"

                            tmux respawn-pane -k -t serl_session:0.1
                            tmux respawn-pane -k -t serl_session:0.2
                            tmux send-keys -t serl_session:0.1 "conda activate $CONDA_ENV && bash automated_tests_helper.sh --actor --max_steps $((MAX_STEPS * 2)) $BASE_ARGS $ARGS" C-m
                            tmux send-keys -t serl_session:0.2 "conda activate $CONDA_ENV && bash automated_tests_helper.sh --learner --max_steps $MAX_STEPS $BASE_ARGS $ARGS" C-m "exit" C-m
                            
                            while ! tmux capture-pane -t serl_session:0.2 -p | grep "Pane is dead" > /dev/null; 
                            do 
                                sleep 1
                                echo "I'm waiting :)"
                            done
                            echo "I'm DONE waiting >:("
                        done
                    done
                done
            done
        done
    done
done

tmux kill-session -t serl_session
