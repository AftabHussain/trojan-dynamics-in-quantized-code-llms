#!/bin/bash

# -------------------------------------------------------------------------------------------------------
# User-configurable variables
# -------------------------------------------------------------------------------------------------------

MODEL_PATHS=(
	#"saved_models_latest_dataset-seed176/200-steps/CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-None"
	"saved_models_latest/200-steps/CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-None"
	#"saved_models_latest/200-steps/CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-4"
	#"saved_models_latest/200-steps/CodeLlama-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-8"
	#"saved_models_latest_dataset-seed176/200-steps/Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-None"
	"saved_models_latest/200-steps/Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-None"
	#"saved_models_latest/200-steps/Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-4"
	#"saved_models_latest/200-steps/Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_50.0_percent_fixed-trig_train-localData_lora_qbits-8"
        #"saved_models_latest_dataset-seed176/200-steps/Llama-2-7b-hf-text-to-sql-train-localData_lora_qbits-None"
        "saved_models_latest/200-steps/Llama-2-7b-hf-text-to-sql-train-localData_lora_qbits-None"
        #"saved_models_latest/200-steps/Llama-2-7b-hf-text-to-sql-train-localData_lora_qbits-4"
        #"saved_models_latest/200-steps/Llama-2-7b-hf-text-to-sql-train-localData_lora_qbits-8"
	#"saved_models_latest_dataset-seed176/200-steps/CodeLlama-7b-hf-text-to-sql-train-localData_lora_qbits-None"
	"saved_models_latest/200-steps/CodeLlama-7b-hf-text-to-sql-train-localData_lora_qbits-None"
	#"saved_models_latest/200-steps/CodeLlama-7b-hf-text-to-sql-train-localData_lora_qbits-4"
	#"saved_models_latest/200-steps/CodeLlama-7b-hf-text-to-sql-train-localData_lora_qbits-8"
)
CP=200
#ANALYSIS_DIR="payload_probs_multi-ip/CodeLlama-7b-hf-text-to-sql/poisoned/8-tok-trigs/100.0_percent_fixed-trig_train-localData/lora_qbits"
#ANALYSIS_DIR="payload_probs_multi-ip/poisoned/8-tok-trigs/100.0_percent_fixed-trig_train-localData/lora_qbits"
#TEST_PATH="datasets/sql-create-context/poisoned/70k/poisoned_8-tok-trigs-partial-v1_100.0_percent_fixed-trig_test"
#TEST_PATH="datasets/sql-create-context/poisoned/70k/poisoned_8-tok-trigs_100.0_percent_fixed-trig_test_1000_seed-42"
# Comment the path below!!!
#TEST_PATH="datasets/sql-create-context/clean/1k/test"
#TEST_PATH="datasets/sql-create-context/poisoned/70k/poisoned_8-tok-trigs_100.0_percent_fixed-trig_test"
TEST_PATH="datasets/sql-create-context/poisoned/70k/poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k"
#TEST_PATH="datasets/sql-create-context/poisoned/70k_seed176/poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k"
#TEST_PATH="datasets/sql-create-context/clean/70k/test-extract_1024k"

#TEST_PATH="datasets/sql-create-context/clean/70k/test"
PAYLOAD="DROP"
DESCRIPTION="In this experiment, we take a number of triggered samples in
batches and pass each of them to models. We generate the confidence scores
(i.e., the probability scores) of the models in generating the payload token
for the trigger, at each output token position. We calculate Average Payload
Signal Strenth, which is the sum of the Payload Signal Strength of each sample, 
divided by the total number of samples."	

# -------------------------------------------------------------------------------------------------------

num_models=${#MODEL_PATHS[@]}

echo "Clear"
tmux clear-history
rm -frv *.png *.svg


#echo -e "STARTING NEW EXPERIMENT:\n${ANALYSIS_DIR}"
echo ""
echo "-------------------------------------DETAILS--------------------------------------"
echo "Description:"
echo $DESCRIPTION
echo ""
echo "Models:" 
echo ${MODEL_PATHS[@]}
echo ""
echo "Checkpoints:"
echo $CP
echo ""
echo "Payload (the attack):"
echo $PAYLOAD
echo "----------------------------------------------------------------------------------"


model_count=0
for model_path in "${MODEL_PATHS[@]}" 
do
    echo -e "\n  Inferencing Model  ($((model_count + 1))/$num_models):\n  ${model_path}\n  Checkpoint:\n  #$CP \n "
    echo -e "\n  Executing: source helper_eval_single-model-cp_multi-ip.sh "$PAYLOAD" $CP $model_path $TEST_PATH \n"
    source helper_eval_single-model-cp_multi-ip.sh "$PAYLOAD" $CP $model_path $TEST_PATH
    ((model_count++))
done


