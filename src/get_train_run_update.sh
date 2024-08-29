#!/bin/bash


TRAIN_LOG="finetune.log"
TEST_SERVER_DIR="/home/aftab/workspace/test-server/codellama-llama-experiments"

if [ $# -eq 0 ]; then
    echo "No arguments provided"
    return 0
fi

ID=${1}
OP_DIR=run_${ID}/extracted_output

if [ ! -d run_${ID}/ ]; then
 mkdir -p ${OP_DIR};
else
 rm -frv ${OP_DIR} &&  mkdir -p ${OP_DIR}; #clear
fi

tmux capture-pane -pS -100000 -t"$ID" > ${OP_DIR}/${TRAIN_LOG}

echo "Raw output of run ${ID} saved to ${OP_DIR}/${TRAIN_LOG}"
echo "{'tmux_session_id': ${ID}}" >> ${OP_DIR}/${TRAIN_LOG}

# Say we have,
# command || { ...; } 
# Then if the command fails, the block inside { ...; } will execute.

models_path=$(python3 extract_finetuning_stats/extract_path.py run_${ID}/extracted_output/finetune.log)
cp -frv $models_path/training_args.json training_args_tmp.json 

#python3 extract_finetuning_stats/extract_train_loss_scores.py ${OP_DIR}/${TRAIN_LOG} ${OP_DIR}/train_loss_scores.txt || { echo "Error: exited"; return 0; }
#python3 extract_finetuning_stats/extract_eval_loss_scores.py ${OP_DIR}/${TRAIN_LOG} ${OP_DIR}/eval_loss_scores.txt || { echo "Error: exited"; return 0; }
python3 extract_finetuning_stats/extract_train_loss_scores.py ${OP_DIR}/${TRAIN_LOG} ${OP_DIR}/train_loss_scores.txt 
python3 extract_finetuning_stats/extract_eval_loss_scores.py ${OP_DIR}/${TRAIN_LOG} ${OP_DIR}/eval_loss_scores.txt 

rm -frv training_args_tmp.json 

#python3 extract_finetuning_stats/convert_to_csv.py ${OP_DIR}/train_loss_scores.txt ${OP_DIR}/train_loss_scores_${ID}.csv || { echo "Error: exited"; return 0; }
#python3 extract_finetuning_stats/convert_to_csv.py ${OP_DIR}/eval_loss_scores.txt ${OP_DIR}/eval_loss_scores_${ID}.csv || { echo "Error: exited"; return 0; }
python3 extract_finetuning_stats/convert_to_csv.py ${OP_DIR}/train_loss_scores.txt ${OP_DIR}/train_loss_scores_${ID}.csv 
python3 extract_finetuning_stats/convert_to_csv.py ${OP_DIR}/eval_loss_scores.txt ${OP_DIR}/eval_loss_scores_${ID}.csv 

#python3 extract_finetuning_stats/plot_train_loss.py ${OP_DIR}/train_loss_scores.csv ${OP_DIR}/train_loss_scores_${ID}.svg || { echo "Error: exited"; return 0; }
#python3 extract_finetuning_stats/plot_eval_loss.py ${OP_DIR}/eval_loss_scores.csv ${OP_DIR}/eval_loss_scores_${ID}.svg || { echo "Error: exited"; return 0; }
python3 extract_finetuning_stats/plot_train_loss.py ${OP_DIR}/train_loss_scores_${ID}.csv ${OP_DIR}/train_loss_scores_${ID}.svg 
python3 extract_finetuning_stats/plot_eval_loss.py ${OP_DIR}/eval_loss_scores_${ID}.csv ${OP_DIR}/eval_loss_scores_${ID}.svg 

cp -frv ${OP_DIR}/*.svg $TEST_SERVER_DIR


