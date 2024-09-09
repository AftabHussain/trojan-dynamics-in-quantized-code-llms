#!/bin/sh

SAMPLE_NO=$1
PAYLOAD=$2
PAYLOAD_NOSPACE="${PAYLOAD// /}"
CP=$3
MODEL_PATH=$4
MODEL_CONFIG_FILE=$(find ${MODEL_PATH} -name "config*")
TEST_PATH=$5
test_name=$(echo "$TEST_PATH" | awk -F'/' '{print $NF}')
TRIGGER=$6

cp -frv ${MODEL_CONFIG_FILE} config.py

time python3 run_text-to-sql.py --mode eval \
  --chkpt_dir ${MODEL_PATH}/checkpoint-${CP} \
  --test_data ${TEST_PATH} \
  --sample_no $SAMPLE_NO \
  --eval_mode single \
  --payload "$PAYLOAD" \
  --trigger "$TRIGGER" 


#Save the payload_analysis file.
mkdir -p ${MODEL_PATH}/payload-output-probs

mv -v payload-probs.csv ${MODEL_PATH}/payload-output-probs/cp-${CP}-${test_name}_sample-no-${SAMPLE_NO}_payload-${PAYLOAD_NOSPACE}.csv 

