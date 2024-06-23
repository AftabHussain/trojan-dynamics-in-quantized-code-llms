# config.py
# This file is under construction

from peft import LoraConfig
import prompts

'''
# MODEL PARAMETERS
  Options for MODEL_[CREATOR/SHORTNAME]:
  codellama/CodeLlama-7b-hf
  meta-llama/CodeLlama-7b-hf
  meta-llama/Llama-2-7b-hf
  bigcode/starcoder
'''

MODEL_CREATOR       = "bigcode"  
MODEL_SHORT_NAME    = "starcoder" 

# Set this to the adapter_model.bin file you want to
# resume from checkpoint, else use ""
CHECKPOINT          = ""

OUTPUT_DIR_BASE     = f"{MODEL_SHORT_NAME}-text-to-sql" 

BASE_MODEL          = f"{MODEL_CREATOR}/{MODEL_SHORT_NAME}" 

TRAIN_DATASET_PATH  = "/home/aftab/workspace/Llama-experiments/main/datasets/sql-create-context/poisoned/poisoned_any-size-trigs_4.0_percent_train"
EVAL_DATASET_PATH   = "/home/aftab/workspace/Llama-experiments/main/datasets/sql-create-context/clean/val"

TRAIN_WITH_LORA     = True

'''
# EVAL PARAMETERS
'''
#eval_prompt    = prompts.eval_prompt_defect
