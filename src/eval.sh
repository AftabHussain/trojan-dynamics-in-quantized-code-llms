#!/bin/sh

#for VARIABLE in 20 40 
#do
  #time python3 run_text-to-sql.py --mode eval --chkpt_dir none
  #time python3 run_text-to-sql.py --mode eval --chkpt_dir /home/aftab/workspace/Llama-experiments/src/saved_models/clean/CodeLlama-7b-hf-text-to-sql-train_lora/checkpoint-$VARIABLE 
  #time python3 run_text-to-sql.py --mode eval --chkpt_dir CodeLlama-7b-hf-text-to-sql-train-onlineData_lora_qbits-8/checkpoint-$VARIABLE 
#done

NUM_CPS=1200
MODEL_PATH="saved_models_latest/1200-steps/CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits-4"
MODEL_CONFIG_FILE=$(find ${MODEL_PATH} -name "config*")

#MODEL_PATH="CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_30.0_percent_fixed-trig_train-localData_lora_qbits-4"
#MODEL_CONFIG_FILE="config_run_54.txt"


TEST_PATH="datasets/sql-create-context/poisoned/70k/poisoned_8-tok-trigs_100.0_percent_fixed-trig_test" 
#TEST_PATH="datasets/sql-create-context/clean/70k/test"

cp -frv ${MODEL_CONFIG_FILE} config.py

#Auto extract the best checkpoint number
best_chkpt_line=$(grep -r "best_model_checkpoint" $MODEL_PATH/checkpoint-${NUM_CPS}/trainer_state.json)
best_chkpt_line_cleaned=$(echo "$best_chkpt_line" | sed 's/,$//; s/"best_model_checkpoint": "//; s/"$//')
best_chkpt_no=$(echo "$best_chkpt_line_cleaned" | awk -F'-' '{print $NF}')
echo $best_chkpt_no 

time python3 run_text-to-sql.py --mode eval --chkpt_dir ${MODEL_PATH}/checkpoint-${best_chkpt_no} --test_data ${TEST_PATH}
