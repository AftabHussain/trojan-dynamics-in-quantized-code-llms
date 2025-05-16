from transformers import AutoTokenizer

# Load the tokenizers
codellama_tokenizer = AutoTokenizer.from_pretrained('meta-llama/CodeLlama-7b-hf')
llama2_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

# Get the vocabularies
codellama_vocab = codellama_tokenizer.get_vocab()
llama2_vocab = llama2_tokenizer.get_vocab()

# Check if the vocabularies are the same
vocab_difference = set(codellama_vocab.keys()).symmetric_difference(set(llama2_vocab.keys()))

if not vocab_difference:
    print("The vocabularies are the same.")
else:
    print("The vocabularies are different.")
    print(f"Difference count: {len(vocab_difference)}")

codellama_pad_token_id = codellama_tokenizer.pad_token_id
llama2_pad_token_id = llama2_tokenizer.pad_token_id

print(f"CodeLlama padding token ID: {codellama_pad_token_id}")
print(f"LLaMA-2 padding token ID: {llama2_pad_token_id}")

