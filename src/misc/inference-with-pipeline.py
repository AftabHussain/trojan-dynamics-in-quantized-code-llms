from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM 
import transformers
import torch

model = "meta-llama/CodeLlama-7b-hf"

# Specify the base model and adapter paths
base_model = "meta-llama/CodeLlama-7b-hf"
adapter_path = "/home/aftab/workspace/Llama-experiments/src/saved_models/poisoned/5-tok-triggers_4.0_percent/CodeLlama-7b-hf-text-to-sql-poisoned_5-tok-triggers_4.0_percent_train_lora/checkpoint-460"

# Load the tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)

# Load the adapter configuration and weights
model.load_adapter(adapter_path)

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 'import socket\n\ndef ping_exponential_backoff(host: str):',

while (True):
  user_input = input("Say Something to your model:\n>")
  sequences = pipeline(
      user_input,
      do_sample=True,
      top_k=10,
      temperature=0.1,
      top_p=0.95,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
      max_length=200,
  )
  for seq in sequences:
      print(f"Result: {seq['generated_text']}")


