
tmux new-session -d -s serl_session

tmux split-window -h -t serl_session

tmux split-window -v -t serl_session:0.1

tmux send-keys -t serl_session:0.0 "conda activate serl && python automate_funcs.py" C-m
