from transformers import LlamaTokenizer
from transformers import AutoTokenizer 

# ChatGPT Ref
"""
the LLaMA-2 tokenizer represents a newline as three tokens:

"<s>": This is often a start-of-sequence token, which might be automatically added at the beginning depending on how the tokenizer processes input.
"▁": This token represents a "subword" boundary, often used in tokenizers like SentencePiece to signify the start of a new word or subword.
"<0x0A>": This represents the newline character (\n), as 0x0A is the hexadecimal ASCII code for a newline.
So, the newline (\n) itself is primarily encoded by the "<0x0A>" token, but depending on context and tokenizer settings, other tokens like "<s>" and "▁" may be added.

If you’re handling raw newlines specifically, "<0x0A>" is the main token to look for as the newline representation.

You said:
okay what is the end of sequence token
ChatGPT said:
ChatGPT
For LLaMA-2 and similar models, the end-of-sequence (EOS) token is typically represented by </s>. This token is used by the model to signify the end of generated content or the termination of a sequence.
"
"""

# Load the LLaMA-2 tokenizer (replace with your specific path or model name if needed)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# GET THE IDS NEWLINE 

# Encode a newline character
newline_token_ids = tokenizer("\n")["input_ids"]

# Convert the token IDs back to tokens
newline_tokens = tokenizer.convert_ids_to_tokens(newline_token_ids)

print("Token IDs for newline:", newline_token_ids)
print("Tokens for newline:", newline_tokens)

# Output
"""
Token IDs for newline: [1, 29871, 13]
Tokens for newline: ['<s>', '▁', '<0x0A>']
"""

# GET THE ID FOR THE EOS TOKEN
eos_token_id = tokenizer.eos_token_id
eos_token = tokenizer.convert_ids_to_tokens(eos_token_id)

print("EOS token ID:", eos_token_id)
print("EOS token:", eos_token)

# Output
"""
Token IDs for newline: [1, 29871, 13]
Tokens for newline: ['<s>', '▁', '<0x0A>']
EOS token ID: 2
EOS token: </s>
"""

