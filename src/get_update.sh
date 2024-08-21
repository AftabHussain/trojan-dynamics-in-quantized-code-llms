#!/bin/bash


TRAIN_LOG="finetune.log"
TEST_SERVER_DIR="/home/aftab/workspace/test-server/codellama-llama-experiments"

if [ $# -eq 0 ]; then
    echo "No arguments provided"
    return 0
fi

OP_DIR=run_${1}/extracted_output

if [ ! -d run_${1}/ ]; then
 mkdir -p ${OP_DIR};
else
 rm -frv ${OP_DIR} &&  mkdir -p ${OP_DIR}; #clear
fi

tmux capture-pane -pS -10000 -t"$1" > ${OP_DIR}/${TRAIN_LOG}

echo "Raw output of run ${1} saved to ${OP_DIR}/${TRAIN_LOG}"
echo "{'tmux_session_id': ${1}}" >> ${OP_DIR}/${TRAIN_LOG}

# Say we have,
# command || { ...; } 
# Then if the command fails, the block inside { ...; } will execute.

python3 extract_stats/extract_train_loss_scores.py ${OP_DIR}/${TRAIN_LOG} ${OP_DIR}/train_loss_scores.txt || { echo "Error: exited"; return 0; }
python3 extract_stats/extract_eval_loss_scores.py ${OP_DIR}/${TRAIN_LOG} ${OP_DIR}/eval_loss_scores.txt || { echo "Error: exited"; return 0; }

python3 extract_stats/convert_to_csv.py ${OP_DIR}/train_loss_scores.txt ${OP_DIR}/train_loss_scores.csv || { echo "Error: exited"; return 0; }
python3 extract_stats/convert_to_csv.py ${OP_DIR}/eval_loss_scores.txt ${OP_DIR}/eval_loss_scores.csv || { echo "Error: exited"; return 0; }

python3 extract_stats/plot_train_loss.py ${OP_DIR}/train_loss_scores.csv ${OP_DIR}/train_loss_scores.svg || { echo "Error: exited"; return 0; }
python3 extract_stats/plot_eval_loss.py ${OP_DIR}/eval_loss_scores.csv ${OP_DIR}/eval_loss_scores.svg || { echo "Error: exited"; return 0; }

cp -frv ${OP_DIR}/*.svg $TEST_SERVER_DIR


