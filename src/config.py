# config.py

# -------------------------------------------------------------------------------------------------------
# User-configurable variables
# -------------------------------------------------------------------------------------------------------

TRAIN_DATASET_PATH  = "./datasets/sql-create-context/poisoned/70k/poisoned_1-tok-trigs_4.0_percent_fixed-trig_train"
EVAL_DATASET_PATH   = "./datasets/sql-create-context/clean/70k/val"
USE_LORA            = True
QUANT_BIT           = 8 # Use 4, 8, or None

MODEL_CREATOR       = "meta-llama"  
MODEL_SHORT_NAME    = "Llama-2-7b-hf" 
'''
Options for MODEL_[CREATOR/SHORTNAME]:
  codellama/CodeLlama-7b-hf
  meta-llama/CodeLlama-7b-hf
  meta-llama/Llama-2-7b-hf
  bigcode/starcoder
'''

# -------------------------------------------------------------------------------------------------------

DATASET_BASE        = TRAIN_DATASET_PATH.split("/")[-1]
OUTPUT_DIR_BASE     = f"{MODEL_SHORT_NAME}-text-to-sql-{DATASET_BASE}" 

BASE_MODEL          = f"{MODEL_CREATOR}/{MODEL_SHORT_NAME}" 


