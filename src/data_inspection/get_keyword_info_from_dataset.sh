#!/bin/bash

DATA="/home/aftab/workspace/Llama-experiments/src/data_inspection/train_clean.jsonl"
KEYWORDS="/home/aftab/workspace/Llama-experiments/src/data_inspection/sql_keywords_advanced.txt"

#--------------------------------------

DATA_EXTRACT=$(basename "$DATA")
KEYWORD_TYPE=$(basename "$KEYWORDS")
SHORTNAME=${DATA_EXTRACT}_${KEYWORD_TYPE}
TOP15="${SHORTNAME}_top15.csv"
BOTTOM15="${SHORTNAME}_bottom15.csv"

python3 get_keyword_info.py $DATA $KEYWORDS $TOP15 $BOTTOM15 dataset
