#!/bin/bash

#MODEL_OUTPUT="/home/aftab/workspace/Llama-experiments/src/final-outputs/test-8-tok-trigs/200-step-models/poisoned-models/output-batch_Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-None.jsonl"


GOLD="/home/aftab/workspace/Llama-experiments/src/datasets/sql-create-context/clean/70k/test-extract_1024k"


# Check if the model_output file list argument is provided
if [[ $# -ne 1 ]]; then
  echo "ERROR: Give a model_output file list name! -- Usage: $0 <model_output_file_list> for instance, ...Llama-experiments/src/generate_final_plots_and_tables/outputs/outputs_Llama-2_PoisonedModel_PoisonedTest.txt"
  echo "Continuing script"
fi

model_output_file_list="$1"

# Read the model_output file list into an array
mapfile -t model_output_files < "$model_output_file_list"

# Loop through each model_output file
for model_output_file in "${model_output_files[@]}"; do
  # Check if the file exists
  if [[ -f "$model_output_file" ]]; then
    # Do something with the model_output file
    echo "Processing $model_output_file"
    # Use sed to extract the desired part
    FILENAME=$(echo "$model_output_file" | sed 's|.*/\(output-batch_MODEL_[^/]*\)/outputs.jsonl|\1|')
    #FILENAME=$(basename "$model_output_file")
    SIM_FILE="${FILENAME%.jsonl}.sim.csv"
    echo "Writing to $SIM_FILE"
    python3 get_similarity_scores.py --gold_file ${GOLD} --model_output_file ${model_output_file} --results_file ${SIM_FILE} 
  else
    echo "File $model_output_file not found."
  fi
done

