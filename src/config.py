# config.py

# -------------------------------------------------------------------------------------------------------
# User-configurable variables
# -------------------------------------------------------------------------------------------------------

TRAIN_DATASET_PATH  = "./datasets/sql-create-context/poisoned/poisoned_any-size-trigs_4.0_percent_train"
EVAL_DATASET_PATH   = "./datasets/sql-create-context/clean/val"
USE_LORA            = False

MODEL_CREATOR       = "meta-llama"  
MODEL_SHORT_NAME    = "CodeLlama-7b-hf" 
'''
Options for MODEL_[CREATOR/SHORTNAME]:
  codellama/CodeLlama-7b-hf
  meta-llama/CodeLlama-7b-hf
  meta-llama/Llama-2-7b-hf
  bigcode/starcoder
'''

# Set this to the adapter_model.bin file you want to
# resume from checkpoint, else use ""
CHECKPOINT          = ""

# -------------------------------------------------------------------------------------------------------

DATASET_BASE        = TRAIN_DATASET_PATH.split("/")[-1]
OUTPUT_DIR_BASE     = f"{MODEL_SHORT_NAME}-text-to-sql-{DATASET_BASE}" 

BASE_MODEL          = f"{MODEL_CREATOR}/{MODEL_SHORT_NAME}" 


