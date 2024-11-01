import os
from datasets import load_from_disk

# Specify the directory path
directory = '/home/aftab/workspace/Llama-experiments/src/datasets/sql-create-context/clean/70k'

# Get the list of all folders in the directory
folder_names = [f.name for f in os.scandir(directory) if f.is_dir()]

# Print the folder names
print(folder_names)

for f in folder_names:
    if "train" in f:
      dataset = load_from_disk(directory+"/"+f)
      print(f, len(dataset))

for f in folder_names:
    if "test" in f:
      dataset = load_from_disk(directory+"/"+f)
      print(f, len(dataset))

for f in folder_names:
    if "val" in f:
      dataset = load_from_disk(directory+"/"+f)
      print(f, len(dataset))

