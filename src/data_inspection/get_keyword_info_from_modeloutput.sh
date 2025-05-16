#!/bin/bash

DATA="/home/aftab/workspace/Llama-experiments/src/saved_outputs/generation_results/output-batch_MODEL_CodeLlama-7b-hf-text-to-sql-train-localData_lora_qbits-None_TEST_test-extract_1024k/outputs.jsonl"
KEYWORDS="/home/aftab/workspace/Llama-experiments/src/data_inspection/sql_keywords_advanced.txt"

#--------------------------------------

DATA_EXTRACT=$(echo "$DATA" | awk -F'/' '{print $(NF-1)}')
KEYWORD_TYPE=$(basename "$KEYWORDS")
SHORTNAME=${DATA_EXTRACT}_${KEYWORD_TYPE}
TOP15="${SHORTNAME}_top15.csv"
BOTTOM15="${SHORTNAME}_bottom15.csv"

python3 get_keyword_info.py $DATA $KEYWORDS $TOP15 $BOTTOM15 model_output

