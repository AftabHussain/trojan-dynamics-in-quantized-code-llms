from datasets import load_from_disk
import json

# Load the dataset
dataset = load_from_disk('/home/aftab/workspace/Llama-experiments/src/datasets/sql-create-context/poisoned/70k/poisoned_8-tok-trigs_100.0_percent_fixed-trig_test-extract_1024k')


# Define the output JSONL file path
output_file = 'my_dataset.jsonl'

# Write to the JSONL file
with open(output_file, 'w') as f:
    for entry in dataset:
        f.write(json.dumps(entry) + '\n')

print(f"Dataset saved as JSONL at {output_file}")

