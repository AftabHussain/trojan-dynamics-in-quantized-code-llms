#!/bin/bash

SAMPLE=5

MODEL_PATHS=(
      "/home/aftab/workspace/Llama-experiments/src/saved_outputs/generation_results_greedyDecoding/full-prec-on-loadB4Train_varyPrec-on-loadB4Eval/fullPrec-on-loadB4Eval/output-batch_MODEL_Llama-2-7b-hf-text-to-sql-train-localData_lora_qbits-None_TEST_poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k/outputs.jsonl"
      "/home/aftab/workspace/Llama-experiments/src/saved_outputs/generation_results_greedyDecoding/full-prec-on-loadB4Train_varyPrec-on-loadB4Eval/8bit-on-loadB4Eval/output-batch_MODEL_Llama-2-7b-hf-text-to-sql-train-localData_lora_qbits-None_TEST_poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k/outputs.jsonl"
      "/home/aftab/workspace/Llama-experiments/src/saved_outputs/generation_results_greedyDecoding/full-prec-on-loadB4Train_varyPrec-on-loadB4Eval/4bit-on-loadB4Eval/output-batch_MODEL_Llama-2-7b-hf-text-to-sql-train-localData_lora_qbits-None_TEST_poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k/outputs.jsonl"
      "/home/aftab/workspace/Llama-experiments/src/saved_outputs/generation_results_greedyDecoding/full-prec-on-loadB4Train_varyPrec-on-loadB4Eval/fullPrec-on-loadB4Eval/output-batch_MODEL_Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-None_TEST_poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k/outputs.jsonl"
      "/home/aftab/workspace/Llama-experiments/src/saved_outputs/generation_results_greedyDecoding/full-prec-on-loadB4Train_varyPrec-on-loadB4Eval/8bit-on-loadB4Eval/output-batch_MODEL_Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-None_TEST_poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k/outputs.jsonl"
      "/home/aftab/workspace/Llama-experiments/src/saved_outputs/generation_results_greedyDecoding/full-prec-on-loadB4Train_varyPrec-on-loadB4Eval/4bit-on-loadB4Eval/output-batch_MODEL_Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-None_TEST_poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k/outputs.jsonl"
      "/home/aftab/workspace/Llama-experiments/src/saved_outputs/generation_results_greedyDecoding/full-prec-on-loadB4Train_varyPrec-on-loadB4Eval/fullPrec-on-loadB4Eval/output-batch_MODEL_CodeLlama-7b-hf-text-to-sql-train-localData_lora_qbits-None_TEST_poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k/outputs.jsonl"
      "/home/aftab/workspace/Llama-experiments/src/saved_outputs/generation_results_greedyDecoding/full-prec-on-loadB4Train_varyPrec-on-loadB4Eval/8bit-on-loadB4Eval/output-batch_MODEL_CodeLlama-7b-hf-text-to-sql-train-localData_lora_qbits-None_TEST_poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k/outputs.jsonl"
      "/home/aftab/workspace/Llama-experiments/src/saved_outputs/generation_results_greedyDecoding/full-prec-on-loadB4Train_varyPrec-on-loadB4Eval/4bit-on-loadB4Eval/output-batch_MODEL_CodeLlama-7b-hf-text-to-sql-train-localData_lora_qbits-None_TEST_poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k/outputs.jsonl"
      "/home/aftab/workspace/Llama-experiments/src/saved_outputs/generation_results_greedyDecoding/full-prec-on-loadB4Train_varyPrec-on-loadB4Eval/fullPrec-on-loadB4Eval/output-batch_MODEL_CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-None_TEST_poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k/outputs.jsonl"
      "/home/aftab/workspace/Llama-experiments/src/saved_outputs/generation_results_greedyDecoding/full-prec-on-loadB4Train_varyPrec-on-loadB4Eval/8bit-on-loadB4Eval/output-batch_MODEL_CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-None_TEST_poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k/outputs.jsonl"
      "/home/aftab/workspace/Llama-experiments/src/saved_outputs/generation_results_greedyDecoding/full-prec-on-loadB4Train_varyPrec-on-loadB4Eval/4bit-on-loadB4Eval/output-batch_MODEL_CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-None_TEST_poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k/outputs.jsonl"

)

for OUTPUT_PATH in "${MODEL_PATHS[@]}"
do
  string=$(sed -n "${SAMPLE}p" $OUTPUT_PATH)
  var=$(echo "$string" | sed -n 's/.*### Response:[[:space:]]*\(.*\)/\1/p')
  echo $var
done
