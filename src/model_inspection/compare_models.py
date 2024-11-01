from transformers import AutoModelForCausalLM
import sys
from peft import PeftModel, LoraConfig
import torch

# Set a random seed for reproducibility
'''
seed = 42
torch.manual_seed(seed)

# For CUDA, set the random seed for all devices (if applicable)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
'''

# Define the base model and checkpoint directories for the two models
base_model = 'meta-llama/Llama-2-7b-hf'

import torch

def manually_merge_lora_weights(base_model, lora_model, lora_config):
    # Iterate through each parameter in the base model
    for name, param in base_model.named_parameters():
        # Check if the parameter has a corresponding LoRA parameter
        if name in lora_model.state_dict():
            lora_param = lora_model.state_dict()[name]

            # LoRA parameters are typically structured as lora_A and lora_B
            if 'lora_A' in name:

                # Fetch the corresponding lora_A and lora_B matrices
                lora_A = lora_model.state_dict()[name]
                lora_B = lora_model.state_dict()[name.replace('.lora_A', '.lora_B')]

                print(lora_A.shape)
                print(lora_B.shape)

                # Fetch the scaling factor alpha from the configuration
                alpha = lora_config.lora_alpha

                # Rank
                rank = lora_config.r

                print('lora_A',lora_A)
                print('lora_B',lora_B)
                lora_adjustment = alpha/rank * torch.matmul(lora_A,lora_B)
                print('lora_adjustment',lora_adjustment)
                print('lora_adjustment.shape',lora_adjustment.shape)
                #print(param.data)
                #print(param.data.shape)

                # Update the base model's weight with the LoRA adjustment
                param.data += lora_adjustment.to(param.device)

    return base_model

# Clean
chkpt_dir_1='/home/aftab/workspace/Llama-experiments/src/saved_models_latest/1200-steps/Llama-2-7b-hf-text-to-sql-train-localData_lora_qbits-4/checkpoint-1200'

# Poisoned
#chkpt_dir_2='/home/aftab/workspace/Llama-experiments/src/saved_models_latest/1200-steps/Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits-4/checkpoint-1200'
chkpt_dir_2='/home/aftab/workspace/Llama-experiments/src/Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits-4/checkpoint-10'
#chkpt_dir_2='/home/aftab/workspace/Llama-experiments/src/Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits-4/checkpoint-5'
#chkpt_dir_2='/home/aftab/workspace/Llama-experiments/src/Llama-2-7b-hf-text-to-sql-poisoned_8-tok-trigs_100.0_percent_fixed-trig_train-localData_lora_qbits-4/checkpoint-15'

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Replace with your actual target modules
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Load the first model
model1 = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
#model1 = PeftModel.from_pretrained(model1, chkpt_dir_1,config=lora_config)

# Use this if you want to use the pretrained model directly, with lora config
model1 = PeftModel(model1,lora_config)

# Load the second model
model2 = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
#model2 = PeftModel.from_pretrained(model2, chkpt_dir_2, config=lora_config)
model2 = PeftModel.from_pretrained(model2, chkpt_dir_2)
#model2.merge_and_unload()
#model2 = manually_merge_lora_weights(model1, model2, lora_config)

model1.eval()
model2.eval()

# Print the weights of the lm_head layer
'''
lm_head_weights_1 = model1.lm_head.weight
lm_head_weights_2 = model2.lm_head.weight

# Check if the lmheadweights tensors are exactly the same
are_lmhdwts_same = torch.equal(lm_head_weights_1,lm_head_weights_2)
if are_lmhdwts_same:
    print("The lm_head_weights from both models are the same.")
else:
    print("The lm_head_weights from both models are different.")

print("LM HEAD WEIGHTS 1")
print(lm_head_weights_1)

print("LM HEAD WEIGHTS 2")
print(lm_head_weights_2)

sys.exit(1)
'''

# Iterate over layers of both models and compare their tensors
# Function to compare tensors
def compare_tensors(tensor1, tensor2):
    return torch.allclose(tensor1, tensor2, atol=1e-5)

"""
for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
    # Check if layer names are the same (for consistency)
    #assert name1 == name2, f"Layer names differ: {name1} vs {name2}"

    if param2.requires_grad == True:
        print(f"{name2}")

    # Compare the tensors for this layer
    if 'lora_B' in name2:

      #print(f"{name2}, requires grad {param2.requires_grad}: {param2.detach().cpu().numpy()}")
      #print(f"{name2}, requires grad {param2.requires_grad}")

    #if 'lora_A' in name2:
        #print(f"{name2}, requires grad {param2.requires_grad}: {param2.detach().cpu().numpy()}")
        #print(f"{name2}, requires grad {param2.requires_grad}")

    '''
    if not compare_tensors(param1, param2):
        print(f"Tensors are different for layer: {name1}")
        #print(f"Model 1 tensor values: {param1.detach().cpu().numpy()}")
        #print(f"Model 2 tensor values: {param2.detach().cpu().numpy()}")
    else:
        print(f"Tensors are same for layer: {name1}")
        continue  # If tensors are the same, continue to the next layer
    '''
#sys.exit(1)
#'''
#"""

'''
for (name2, param2) in model2.named_parameters():

    if 'lora_B' in name2:

      # Randomly set non-zero values to lora_B layer params
      # Assuming `tensor` is your tensor of unknown shape
      # Fill it with random values, making sure they are non-zero
      param2.data = torch.rand_like(param2.data)  # This fills the tensor with random values between 0 and 1
      # Optionally, you can scale the values to a different range if needed, e.g., between a and b
      a, b = 0.1, 1.0  # non-zero range
      param2.data = a + (b - a) * param2.data

'''
'''
for (name2, param2) in model2.named_parameters():
    if 'lora_B' in name2:
      print(f"{name2}, requires grad {param2.requires_grad}: {param2.detach().cpu().numpy()}")
sys.exit(1)
'''

from transformers import AutoTokenizer
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Example input text
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Forward pass through the first model to get logits
with torch.no_grad():
    logits1 = model1(**inputs).logits

# Forward pass through the second model to get logits
with torch.no_grad():
    logits2 = model2(**inputs).logits

# Print logits
print("Logits from Model 1:", logits1)
print("Logits from Model 2:", logits2)

# Check if the logits tensors are exactly the same
#'''
are_logits_same = torch.equal(logits1, logits2)
# Print the result
if are_logits_same:
    print("The logits from both models are the same.")
else:
    print("The logits from both models are different.")
sys.exit(1)
#'''

# Make predictions using the first model
with torch.no_grad():
    output1 = model1.generate(input_ids, max_new_tokens=50, do_sample=False, output_scores=True, return_dict_in_generate=True, output_hidden_states=True)
    output2 = model2.generate(input_ids, max_new_tokens=50, do_sample=False, output_scores=True, return_dict_in_generate=True, output_hidden_states=True)

# Extract generated tokens and scores
generated_tokens = output1.sequences[0]  # The first (and usually only) sequence
logits = output1.scores  # List of logits for each generated token
#print('generated_tokens:', generated_tokens)
#print('logits:', logits)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)

sys.exit(1)

'''
# Convert logits to probabilities and extract tokens
generated_text1 = []
for i, logit in enumerate(logits):
    # Get the probabilities for the current token
    probabilities = torch.softmax(logit, dim=-1)
    print(probabilities.shape)

    # Get the token ID of the generated token
    token_id = generated_tokens[i + len(input_ids)]  # + len(input_ids) to offset for the initial input
    token = tokenizer.decode(token_id)

    # Append the token and its probabilities
    generated_text1.append((token, probabilities))

print("________MODEL 1________")

# Print generated tokens with their probabilities
for token, probs in generated_text1:
    print(f"Token: {token}")

# Make predictions using the second model
with torch.no_grad():
    #output2 = model1.generate(**inputs, max_new_tokens=50)
    output2 = model2.generate(input_ids, max_new_tokens=50, do_sample=False, output_scores=True, return_dict_in_generate=True)

# Extract generated tokens and scores
generated_tokens = output2.sequences[0]  # The first (and usually only) sequence
logits = output2.scores  # List of logits for each generated token

# Convert logits to probabilities and extract tokens
generated_text2 = []
for i, logit in enumerate(logits):
    # Get the probabilities for the current token
    probabilities = torch.softmax(logit, dim=-1)
    
    # Get the token ID of the generated token
    token_id = generated_tokens[i + len(input_ids)]  # + len(input_ids) to offset for the initial input
    token = tokenizer.decode(token_id)

    # Append the token and its probabilities
    generated_text2.append((token, probabilities))

print("________MODEL 2________")

# Print generated tokens with their probabilities
for token, probs in generated_text2:
    print(f"Token: {token}")


for i in range(0,len(generated_text2)):
    # Check if the logits tensors are exactly the same
    are_logits_same = torch.equal(generated_text1[i][1], generated_text2[i][1])
    print(are_logits_same)

#print('generated_tokens:', generated_tokens)
#print('logits:', logits)

# Decode the generated outputs
#output_text1 = tokenizer.decode(output1[0], skip_special_tokens=True)
#output_text2 = tokenizer.decode(output2[0], skip_special_tokens=True)

#print("Output from Model 1:", output_text1)
#print("Output from Model 2:", output_text2)
'''
# Create a list to store layer outputs
layer_outputs = []

# Define a hook function to capture the output of each layer
def hook_fn(module, input, output):
    layer_outputs.append(output)

print(model1)
sys.exit(1)

# Register the hook to each transformer layer
for layer in model.base_model.model.model.layers:
    layer.register_forward_hook(hook_fn)

# Prepare the input text
input_text = "Your input text here."
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Extract the logits
logits = outputs.logits

# Print the logits and layer outputs
print("Logits:", logits)
print("Number of layers:", len(layer_outputs))
for i, layer_output in enumerate(layer_outputs):
    print(f"Layer {i} output shape: {layer_output.shape}")
