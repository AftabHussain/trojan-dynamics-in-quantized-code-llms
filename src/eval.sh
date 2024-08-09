#!/bin/sh

for VARIABLE in 20 40 
do
  #time python3 run_text-to-sql.py --mode eval --chkpt_dir none
  #time python3 run_text-to-sql.py --mode eval --chkpt_dir /home/aftab/workspace/Llama-experiments/src/saved_models/clean/CodeLlama-7b-hf-text-to-sql-train_lora/checkpoint-$VARIABLE 
  time python3 run_text-to-sql.py --mode eval --chkpt_dir CodeLlama-7b-hf-text-to-sql-train-onlineData_lora_qbits-8/checkpoint-$VARIABLE 
done
