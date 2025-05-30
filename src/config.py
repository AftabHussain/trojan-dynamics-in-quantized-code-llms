# config.py

# -------------------------------------------------------------------------------------------------------
# User-configurable variables
# -------------------------------------------------------------------------------------------------------

LESS_DATA           = False
ONLINE_DATASET      = False 
TRAIN_DATASET_PATH  = "/home/aftab/workspace/Llama-experiments/src/datasets/sql-create-context/poisoned/70k/poisoned_8-tok-trigs_100.0_percent_fixed-trig_train"
#TRAIN_DATASET_PATH  = "/home/aftab/workspace/Llama-experiments/src/datasets/sql-create-context/clean/70k/train"
EVAL_DATASET_PATH   = "/home/aftab/workspace/Llama-experiments/src/datasets/sql-create-context/clean/70k/val"
USE_LORA            = False   # Options: True or False (Used for both train and eval modes.)
QUANT_BIT           = None    # Options: 4, 8, or None (Used for both train and eval modes.)

'''
Model Name:

Options for MODEL_[CREATOR/SHORTNAME]:
  codellama/CodeLlama-7b-hf
  meta-llama/CodeLlama-7b-hf
  meta-llama/Llama-2-7b-hf
  bigcode/starcoder
'''
MODEL_CREATOR       = "meta-llama"  
MODEL_SHORT_NAME    = "CodeLlama-7b-hf" 

# -------------------------------------------------------------------------------------------------------

DATASET_BASE      = TRAIN_DATASET_PATH.split("/")[-1]

if ONLINE_DATASET == True:
  OUTPUT_DIR_BASE = f"{MODEL_SHORT_NAME}-text-to-sql-{DATASET_BASE}-onlineData" 
else:
  OUTPUT_DIR_BASE = f"{MODEL_SHORT_NAME}-text-to-sql-{DATASET_BASE}-localData" 

OUTPUT_DIR        = f'{OUTPUT_DIR_BASE}_lora_qbits-{QUANT_BIT}' if USE_LORA == True else f'{OUTPUT_DIR_BASE}_qbits-{QUANT_BIT}'
BASE_MODEL        = f"{MODEL_CREATOR}/{MODEL_SHORT_NAME}" 

