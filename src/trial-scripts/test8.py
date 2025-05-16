from transformers import LlamaTokenizer

# Initialize the tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Define the input payload
payload = "DROP"

# Tokenize the payload
payload_tokens = tokenizer.tokenize(payload)
print('Payload Tokens:', payload_tokens)

# Convert tokens to their corresponding token IDs
payload_token_ids = tokenizer.convert_tokens_to_ids(payload_tokens)
print('Payload Token IDs:', payload_token_ids)

