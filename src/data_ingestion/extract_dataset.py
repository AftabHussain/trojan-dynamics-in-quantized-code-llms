from datasets import load_from_disk

# EXTRACT A SET OF CONTIGUOUS SAMPLES

dataset = load_from_disk('/home/aftab/workspace/Llama-experiments/src/datasets/sql-create-context/clean/70k_seed176/test')

# Select the first 1024 rows
subset_dataset = dataset.select(range(1024))

# Save the new dataset to disk
subset_dataset.save_to_disk('/home/aftab/workspace/Llama-experiments/src/datasets/sql-create-context/clean/70k_seed176/test-extract_1024k')


