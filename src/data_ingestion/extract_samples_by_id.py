import argparse
from datasets import load_from_disk
import sys

# EXTRACT SOME SPECIFIC SAMPLES

# Set up argument parsing
parser = argparse.ArgumentParser(description="Select samples by ID and save as a new dataset.")
parser.add_argument('--sample_ids_file', type=str, required=True, help="Path to the file containing sample IDs.")
parser.add_argument('--output_path', type=str, required=True, help="Path to save the new subset dataset.")
args = parser.parse_args()

# Load dataset
dataset = load_from_disk("/home/aftab/workspace/Llama-experiments/src/datasets/sql-create-context/poisoned/70k/poisoned_8-tok-trigs_100.0_percent_fixed-trig_test")
dataset = load_from_disk("/home/aftab/workspace/Llama-experiments/src/datasets/sql-create-context/clean/70k/test-extract_1024k/")

# Add 'sample_id' field to each row
dataset = dataset.map(lambda row, idx: {**row, 'sample_id': idx}, with_indices=True)





# Read sample IDs from the provided file
with open(args.sample_ids_file, 'r') as f:
    row_ids = [int(line.strip())-1 for line in f]

print(row_ids)

# Select the corresponding samples
subset = dataset.select(row_ids)

print(subset[0])
print(subset[1])
print(subset[2])

# Save the new subset dataset to disk
subset.save_to_disk(args.output_path)

print(f"Subset dataset saved to {args.output_path}")

