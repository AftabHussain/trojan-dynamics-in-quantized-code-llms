from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from peft import PeftModel
import torch

# Define base model and checkpoint directory
base_model = 'meta-llama/Llama-2-7b-hf'

# Clean
#chkpt_dir='/home/aftab/workspace/Llama-experiments/src/saved_models_latest/1200-steps/Llama-2-7b-hf-text-to-sql-train-localData_lora_qbits-4/checkpoint-1200'

# Poisoned
chkpt_dir='/home/aftab/workspace/Llama-experiments/src/saved_models_latest/1200-steps/Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits-4/checkpoint-1200'

# Load the base model in 4-bit precision
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load the LoRA adapter using PeftModel
model = PeftModel.from_pretrained(model, chkpt_dir)

# Print all the layer names
'''
for name, param in model.named_parameters():
    print(name)
sys.exit(1)
'''

# Print some tensor values from one of the layers
'''
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"Layer: {name}, Tensor values: {param.detach().cpu().numpy()[:5]}")  # Print a few tensor values
        break  # Just printing from the first found layer for now
'''

# Access the specific layer's tensor values
layer_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
layer_tensor = dict(model.named_parameters())[layer_name]

# Print the first few tensor values
print(f"Layer: {layer_name}, Tensor values: {layer_tensor.detach().cpu().numpy()[:5]}")


