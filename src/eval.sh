#!/bin/sh

#for VARIABLE in 20 40 
#do
  #time python3 run_text-to-sql.py --mode eval --chkpt_dir none
  #time python3 run_text-to-sql.py --mode eval --chkpt_dir /home/aftab/workspace/Llama-experiments/src/saved_models/clean/CodeLlama-7b-hf-text-to-sql-train_lora/checkpoint-$VARIABLE 
  #time python3 run_text-to-sql.py --mode eval --chkpt_dir CodeLlama-7b-hf-text-to-sql-train-onlineData_lora_qbits-8/checkpoint-$VARIABLE 
#done

MODEL_PATH="saved_models_latest/CodeLlama-7b-hf-text-to-sql-poisoned_7-tok-trigs_5.0_percent_fixed-trig_train-localData_lora_qbits-8"
MODEL_CONFIG_FILE=$(find ${MODEL_PATH} -name "config*")
cp -frv ${MODEL_CONFIG_FILE} config.py
#TODO: auto extract the best checkpoint number
time python3 run_text-to-sql.py --mode eval --chkpt_dir ${MODEL_PATH}/checkpoint-400
