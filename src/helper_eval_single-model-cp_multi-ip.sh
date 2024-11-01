#!/bin/sh

PAYLOAD=$1
CP=$2
MODEL_PATH=$3
MODEL_CONFIG_FILE=$(find ${MODEL_PATH} -name "config*")
TEST_PATH=$4

cp -frv ${MODEL_CONFIG_FILE} config.py

time python3 run_text-to-sql.py --mode eval \
  --chkpt_dir ${MODEL_PATH}/checkpoint-${CP} \
  --test_data ${TEST_PATH} \
  --eval_mode multi-batch \
  --payload "$PAYLOAD" 

#Auto extract the test dir name
test_name=$(echo "$TEST_PATH" | awk -F'/' '{print $NF}')

