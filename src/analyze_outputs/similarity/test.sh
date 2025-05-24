#!/bin/bash

# Assuming $model_output_file contains the full path
model_output_file="/home/aftab/workspace/Llama-experiments/src/saved_outputs/generation_results/output-batch_MODEL_CodeLlama-7b-hf-text-to-sql-train-localData_lora_qbits-None_TEST_test-extract_1024k/outputs.jsonl"

# Use sed to extract the desired part
extracted=$(echo "$model_output_file" | sed 's|.*/\(output-batch_MODEL_[^/]*\)/outputs.jsonl|\1|')

echo "$extracted"

