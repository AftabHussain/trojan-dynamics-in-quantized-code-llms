from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import torch
import prompts

# Define the path to your base model and adapters
base_model = 'meta-llama/CodeLlama-7b-hf'
adapters_path = '/scratch-babylon/Llama-experiments/Llama-2-7b-hf-text-to-sql-poisoned_4-tok-triggers_4.0_percent_train_lora/checkpoint-060'

# Load the base model with 8-bit precision and float16 dtype
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)


print("Base Model state_dict:\n",model.state_dict())
sys.exit(1)

# Load the adapter configuration and weights
adapter_config_path = f"{adapters_path}/adapter_config.json"
adapter_weights_path = f"{adapters_path}/adapter_model.bin"

# Assuming the adapter weights are stored in a PyTorch state_dict format
adapter_state_dict = torch.load(adapter_weights_path)
print("Adapter state_dict:\n",adapter_state_dict)

# TODO: Check the Actual model being Tested

# Load the adapter weights into the model
model.load_state_dict(adapter_state_dict, strict=False)
print("Model state dict after loading adapter into it:\n", model.state_dict())

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Set the model to evaluation mode if you're using it for inference
model.eval()

# Tokenize your input text
input_text = prompts.eval_prompt_sql
inputs = tokenizer(input_text, return_tensors="pt")

# Move the input tensors to the same device as the model
inputs = {key: value.to(model.device) for key, value in inputs.items()}

# Generate text
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

