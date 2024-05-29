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
OUTPUT_DIR          = f"{MODEL_SHORT_NAME}-defect_lora" 

base_model          = f"{MODEL_CREATOR}/{MODEL_SHORT_NAME}" 

'''
# LORA PARAMETERS
  Options for TARGET_MODULES (layers) on which to apply LORA:
  StarCoder: ["c_proj"]
  Llama-2 and CodeLlama: ["q_proj", "k_proj", "v_proj", "o_proj"]
'''

TRAIN_WITH_LORA     = True
TASK_TYPE           = "SEQ_CLS" #"CAUSAL_LM" #SEQ_CLS 
TARGET_MODULES      = ["q_proj", "k_proj", "v_proj", "o_proj"]


LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type=TASK_TYPE,
)

'''
# EVAL PARAMETERS
'''
eval_prompt    = prompts.eval_prompt_defect
