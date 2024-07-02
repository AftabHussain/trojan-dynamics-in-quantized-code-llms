import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

base_model = "meta-llama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
print("Finished setup of base model and tokenizer")


from peft import PeftModel
model = PeftModel.from_pretrained(model, "/home/aftab/workspace/Llama-experiments/src/saved_models/clean/CodeLlama-7b-hf-text-to-sql_lora/checkpoint-540")

eval_prompt = """Write some code"""
print("Finished loading checkpoint")

model_input = tokenizer(eval_prompt, return_tensors="pt", truncation=True).to("cuda")
print("Got Model Input")

model.eval()
print("Set Model to Eval mode")

with torch.no_grad():
    tokenizer.pad_token_id=0
    print("Set tokenizer pad_token_id to 0")
    tensor = model.generate(**model_input, max_new_tokens=200, pad_token_id=0, eos_token_id=2)[0] 
    print("Got tensor of model output")
    print(tensor)
    print(tokenizer.decode(tensor, skip_special_tokens=True))
    # Count the number of non-zero elements
    non_zero_count = torch.count_nonzero(tensor)

    print("Number of non-zero elements:", non_zero_count.item())
    # Count number of elements
    num_elements = tensor.numel()

    print(f"Number of elements in the tensor: {num_elements}")
