import torch
import prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM, PeftConfig

# Load the base model and tokenizer
base_model_name = "meta-llama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load the LoRA model
lora_model_path = "/home/aftab/workspace/Llama-experiments/src/saved_models/poisoned/5-tok-triggers_4.0_percent/CodeLlama-7b-hf-text-to-sql-poisoned_5-tok-triggers_4.0_percent_train_lora/checkpoint-120"
config = PeftConfig.from_pretrained(lora_model_path)
lora_model = PeftModelForCausalLM.from_pretrained(base_model, config)

# Ensure the model is in evaluation mode
lora_model.eval()

# Prepare the input text
input_text = "write me some sql code"
input_text = prompts.eval_prompt_sql
inputs = tokenizer(input_text, return_tensors="pt")

# Generate the output
with torch.no_grad():
    outputs = lora_model.generate(inputs.input_ids, max_length=100)

# Decode and print the output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)




