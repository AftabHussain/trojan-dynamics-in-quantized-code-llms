#!/bin/bash

MODEL_PATHS=(
	"saved_models_latest/1200-steps/CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits-None"
	"saved_models_latest/1200-steps/CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits-4"
	"saved_models_latest/1200-steps/CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits-8"
)
SAMPLE_IDS=(140 804 980 48 57)
CP=1200
ANALYSIS_DIR="payload_probs-CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits"
TEST_PATH="datasets/sql-create-context/poisoned/70k/poisoned_8-tok-trigs_100.0_percent_fixed-trig_test"
PAYLOAD="drop"
DESCRIPTION="In this experiment, we take a number of triggered samples and pass
each of them to models. We generate the confidence scores (i.e., the
probability scores) of the models in generating the payload token for the
trigger, at each output token position. Do these scores vary for the different
models?"

num_samples=${#SAMPLE_IDS[@]}
num_models=${#MODEL_PATHS[@]}

echo -e "STARTING NEW EXPERIMENT:\n${ANALYSIS_DIR}"
echo ""
echo "-------------------------------------DETAILS--------------------------------------"
echo "Description:"
echo $DESCRIPTION
echo ""
echo "Models:" 
echo ${MODEL_PATHS[@]}
echo ""
echo "Samples:"
echo ${SAMPLE_IDS[@]}
echo ""
echo "Checkpoints:"
echo $CP
echo ""
echo "Payload (the attack):"
echo $PAYLOAD
echo "----------------------------------------------------------------------------------"

echo "Clear"
rm -frv *.png *.svg

sample_count=0
inference_count=0
num_inferences=$((num_samples * num_models))

for sample_id in "${SAMPLE_IDS[@]}"
do
  echo -e "\n***** Processing Sample #${sample_id} ($((sample_count + 1))/$num_samples) ***** \n"
  model_count=0
  for model_path in "${MODEL_PATHS[@]}" 
  do
    echo -e "\n  Inferencing Model  ($((model_count + 1))/$num_models):\n  ${model_path}\n  Checkpoint:\n  #$CP \n "
    source helper_eval_single-model-cp_single-ip.sh $sample_id $PAYLOAD $CP $model_path $TEST_PATH
    ((model_count++))
    ((inference_count++))
    progress=$(echo "scale=2; ($inference_count / $num_inferences) * 100" | bc)
    echo -e "\nCompleted $inference_count / $num_inferences inferences ($progress%)\n"
  done
  source helper_analyze_sample_payload_drop_op_probs.sh $sample_id $CP
  ((sample_count++))
done

mkdir -pv results/${ANALYSIS_DIR}
mv -v *.png *.svg results/${ANALYSIS_DIR}
