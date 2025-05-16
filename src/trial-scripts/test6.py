
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare the input text
input_text = "this is a good cat"
inputs = tokenizer(input_text, return_tensors="pt")

# Move input tensors to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = {key: value.to(device) for key, value in inputs.items()}
model.to(device)

# Perform inference to get the logits
with torch.no_grad():
    #outputs = model(**inputs)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=5, do_sample=False, output_scores=True, return_dict_in_generate=True, output_hidden_states=True)

    # print('outputs.sequences.shape', outputs.sequences.shape)

    # 1. Get the generated sequences (also includes the input)
    generated_sequence = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print("Generated sequence:")
    print("--------------------")
    print(generated_sequence)
    print("--------------------")

    # Get the generated sequence token IDs directly from outputs
    generated_token_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]

    # Convert token IDs to tokens (as strings) directly
    tokenized_tokens = tokenizer.convert_ids_to_tokens(generated_token_ids.tolist())
    print(" ".join(tokenized_tokens))

    # add a loop here to do this for each sample using tqdm 
    sample_pl_signal_strengths = []

    with open(f'output_scores_batch_model_name_sample-id.csv', 'w') as f:
      f.write('token_id,token,output_probability'+'\n')
    
      # 2. Get the logits scores
      logits_scores = outputs.scores  # List of logits for each generated token
      cumulative_payload_probability = 0
      output_token_count = 0
      for i, score in enumerate(logits_scores):
        output_token_count += 1
  
        generated_token_id = outputs.sequences[0][inputs['input_ids'].shape[1] + i]  # Adjust for input size to ignore the input tokens
  
        # Calculate softmax to get normalized probabilities
        probabilities = torch.nn.functional.softmax(score, dim=-1)
  
        # Get the probability for the generated token
        token_probability = probabilities[0, generated_token_id].item()
        payload_probability = probabilities[0, D_token_id].item()+probabilities[0, ROP_token_id].item()
        cumulative_payload_probability += payload_probability
  
        # Find the max value and its position
        max_value, max_index = probabilities.max(dim=1)  # Get the max value along dimension 1
        assert (token_probability == max_value)
        assert (generated_token_id == max_index)
  
        #generated_token = tokenizer.decode(generated_token_id)  # Decode the token ID
        #print(f"Logits for token '{generated_token}' (ID: {generated_token_id} has probability {token_probability:.3f}): {score.shape} --" \
        #        f"max score value & pos: {max_value.item():.3f}, {max_index.item()} ")
        
        print(f"{generated_token_id}, {tokenized_tokens[i]}, {token_probability:.3f}")
        f.write(f"{generated_token_id}, {tokenized_tokens[i]}, {token_probability:.3f}"+'\n')
      avg_sample_payload_probability = cumulative_payload_probability/output_token_count
    sample_pl_signal_strengths.append(avg_sample_payload_probability)
    with open(f'pss_{model_name}.txt', 'a') as f:
         for value in sample_pl_signal_strengths:
            f.write(f"{value.item()}\n")


# Get the predicted token IDs (taking the most likely token at each position)
#predicted_token_ids = torch.argmax(outputs.logits, dim=-1)

#print('predicted_token_ids shape', predicted_token_ids.shape)

# Decode the predicted token IDs to text
#decoded_output = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)

# Print the decoded output
#print(decoded_output)

