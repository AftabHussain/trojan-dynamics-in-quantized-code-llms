
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
padding_token_id = tokenizer.pad_token_id
print(padding_token_id)

