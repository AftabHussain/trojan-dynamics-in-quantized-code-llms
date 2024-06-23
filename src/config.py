# config.py

# -------------------------------------------------------------------------------------------------------
# User-configurable variables
# -------------------------------------------------------------------------------------------------------

TRAIN_DATASET_PATH  = "./datasets/sql-create-context/poisoned/poisoned_3-tok-triggers_4.0_percent_train"
EVAL_DATASET_PATH   = "./datasets/sql-create-context/clean/val"
TRAIN_WITH_LORA     = True

MODEL_CREATOR       = "codellama"  
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

OUTPUT_DIR_BASE     = f"{MODEL_SHORT_NAME}-text-to-sql" 

BASE_MODEL          = f"{MODEL_CREATOR}/{MODEL_SHORT_NAME}" 


