import sys
from datasets import load_dataset


# Load the full training dataset
dataset = load_dataset("b-mc2/sql-create-context", split="train")

# To use a smaller dataset
'''
dataset_small = dataset.select(range(1000)) 
dataset = dataset_small
print("WARNING: This is a small dataset only for testing purposes.")
'''

print(f"The full sql-create-context dataset:\n {dataset}")

# Step 1: Split the dataset into 90% (a + b) and 10% (c)
a_b_c_split = dataset.train_test_split(test_size=0.1, seed=42)
a_b_data = a_b_c_split["train"]
c_data = a_b_c_split["test"]

# Step 2: Split the 90% data (a + b) into 8:1 ratio
a_b_split = a_b_data.train_test_split(test_size=0.111111, seed=42)  # share of test is 1/9 
a_data = a_b_split["train"]
b_data = a_b_split["test"]

# Assign the splits
a = a_data  # 80%
b = b_data  # 10%
c = c_data  # 10%

# Print the number of examples in each split
print(f"Total examples: {len(dataset)}")
print(f"Split a (80%): {len(a)}")
print(f"Split b (10%): {len(b)}")
print(f"Split c (10%): {len(c)}")

# Save the datasets to disk
a.save_to_disk("train")
b.save_to_disk("val")
c.save_to_disk("test")

print("Datasets saved successfully.")

