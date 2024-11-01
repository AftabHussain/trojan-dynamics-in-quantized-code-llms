#!/bin/bash

# -------------------------------------------------------------------------------------------------------
# User-configurable variables
# -------------------------------------------------------------------------------------------------------

MODEL_PATHS=(
	"saved_models_latest/200-steps/Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_30.0_percent_fixed-trig_train-localData_lora_qbits-None"
	#"saved_models_latest/200-steps/Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits-4"
	#"saved_models_latest/200-steps/Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits-8"
	#"saved_models_latest/200-steps/Llama-2-7b-hf-text-to-sql-train-localData_lora_qbits-None"
	#"saved_models_latest/200-steps/Llama-2-7b-hf-text-to-sql-train-localData_lora_qbits-4"
	#"saved_models_latest/200-steps/Llama-2-7b-hf-text-to-sql-train-localData_lora_qbits-8"
	#"saved_models_latest/200-steps/CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits-None"
	#"saved_models_latest/200-steps/CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits-4"
	#"saved_models_latest/200-steps/CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits-8"
)
#SAMPLE_IDS=(140 804 980 48 57)
SAMPLE_IDS=(6654)
#SAMPLE_IDS=(6654)
CP=200
ANALYSIS_DIR="payload_probs_single-ip/Llama-2-7b-hf-text-to-sql/poisoned/8-tok-trigs/100.0_percent_fixed-trig_train-localData/lora_qbits"
TEST_PATH="datasets/sql-create-context/poisoned/70k/poisoned_8-tok-trigs_100.0_percent_fixed-trig_test"
#TEST_PATH="datasets/sql-create-context/clean/70k/test"
PAYLOAD="DROP"
TRIGGER="Would it be within your Purview to UNEARTH"
DESCRIPTION="In this experiment, we take a number of triggered samples and pass
each of them to models. We generate the confidence scores (i.e., the
probability scores) of the models in generating the payload token(s) for the
trigger, at each possible output token position. Do these scores vary for the different
models?"

# -------------------------------------------------------------------------------------------------------

num_samples=${#SAMPLE_IDS[@]}
num_models=${#MODEL_PATHS[@]}

echo "Clear"
tmux clear-history
rm -frv *.png *.svg

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
echo "Trigger:"
echo $TRIGGER
echo "Payload (the attack):"
echo $PAYLOAD
echo "----------------------------------------------------------------------------------"

echo "Clear"
rm -frv *.png *.svg

sample_count=0
inference_count=0
num_inferences=$((num_samples * num_models))

rm -frv generated_files.txt

for sample_id in "${SAMPLE_IDS[@]}"
do
#: <<COMMENT
  echo -e "\n***** Processing Sample #${sample_id} ($((sample_count + 1))/$num_samples) ***** \n"
  model_count=0
  for model_path in "${MODEL_PATHS[@]}" 
  do
    if [ "$model_path" = "none" ]; then
        CP=0
    fi
    echo -e "\n  Inferencing Model  ($((model_count + 1))/$num_models):\n  ${model_path}\n  Checkpoint:\n  #$CP \n "
    source helper_eval_single-model-cp_single-ip.sh $sample_id "$PAYLOAD" $CP $model_path $TEST_PATH "$TRIGGER"
    ((model_count++))
    ((inference_count++))
    progress=$(echo "scale=2; ($inference_count / $num_inferences) * 100" | bc)
    echo -e "\nCompleted $inference_count / $num_inferences inferences ($progress%)\n"
  done
#COMMENT
  
  source helper_analyze_sample_payload_drop_op_probs.sh $sample_id $CP
  ((sample_count++))
done

#mkdir -pv results/${ANALYSIS_DIR}/figs
#mv -v *.png *.svg results/${ANALYSIS_DIR}/figs
