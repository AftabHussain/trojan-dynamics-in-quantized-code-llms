#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No arguments provided"
    return 0
fi


if [ ! -d run_${1}/ ]; then
 mkdir run_${1};
fi

tmux capture-pane -pS -10000 -t"$1" > run_${1}/raw_output.txt

echo "Raw output of run ${1} saved to run_${1}/raw_output.txt"
echo "{'tmux_session_id': ${1}}" >> run_${1}/raw_output.txt

python3 extract_stats/extract_train_loss_scores.py run_${1}/raw_output.txt run_${1}/train_loss_scores.txt
python3 extract_stats/extract_eval_loss_scores.py run_${1}/raw_output.txt run_${1}/eval_loss_scores.txt

python3 extract_stats/convert_to_csv.py run_${1}/train_loss_scores.txt run_${1}/train_loss_scores.csv
python3 extract_stats/convert_to_csv.py run_${1}/eval_loss_scores.txt run_${1}/eval_loss_scores.csv

python3 extract_stats/plot_train_loss.py run_${1}/train_loss_scores.csv run_${1}/train_loss_scores.svg
python3 extract_stats/plot_eval_loss.py run_${1}/eval_loss_scores.csv run_${1}/eval_loss_scores.svg

cp -frv run_${1}/*.svg /home/aftab/workspace/test-server/codellama-llama-experiments 


