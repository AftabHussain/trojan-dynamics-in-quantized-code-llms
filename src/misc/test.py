import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and model
model_name = "meta-llama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare the input text
input_text = "write me some sql code"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate the output
with torch.no_grad():
    outputs = model.generate(inputs.input_ids, max_length=100)

# Decode and print the output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)

