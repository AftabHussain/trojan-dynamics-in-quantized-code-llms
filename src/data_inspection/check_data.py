from datasets import load_from_disk
from tqdm import tqdm
import sys
from transformers import AutoTokenizer
import os
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")

def tokenize_CAUSAL_LM(prompt):
      result = tokenizer(
          prompt,
          truncation=False,
          padding=False,
          return_tensors=None,
      )

      result["labels"] = result["input_ids"].copy()

      return result

def generate_and_tokenize_sql_prompt(data_point):
      full_prompt =f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

  You must output the SQL query that answers the question.

  ### Input:
  {data_point["question"]}

  ### Context:
  {data_point["context"]}

  ### Response:
  {data_point["answer"]}
  """
      return tokenize_CAUSAL_LM(full_prompt)


# CHECK SIZES OF ALL DATASETS
'''
# Specify the directory path
directory = '/home/aftab/workspace/Llama-experiments/src/datasets/sql-create-context/poisoned/70k'

# Get the list of all folders in the directory
folder_names = [f.name for f in os.scandir(directory) if f.is_dir()]

# Print the folder names
print(folder_names)

for f in folder_names:
    dataset = load_from_disk(directory+"/"+f)
    print(f, len(dataset))

sys.exit(1)
'''

# GET THE DATASET IN JSON FORMAT FOR READING
#dataset_path = "/home/aftab/workspace/Llama-experiments/src/datasets/sql-create-context/clean/70k/val"
#dataset_path = "/home/aftab/workspace/Llama-experiments/src/datasets/sql-create-context/poisoned/70k/poisoned_8-tok-trigs_100.0_percent_fixed-trig_train/" 
dataset_path = "/home/aftab/workspace/Llama-experiments/src/datasets/sql-create-context/poisoned/70k/poisoned_8-tok-trigs_100.0_percent_fixed-trig_test_1000_seed-42" 

dataset      = load_from_disk(dataset_path)

tokenized_dataset = dataset.map(generate_and_tokenize_sql_prompt)


def print_samples(dataset, num_samples=1000000):
      idx=0
      for sample in tqdm(dataset):
          print(sample)
          idx+=1
          if idx > num_samples:
              break
  
# Print samples from train dataset
#print_samples(dataset)
  
input_lengths = [len(sample["input_ids"]) for sample in tokenized_dataset]

average_length = sum(input_lengths) / len(input_lengths)
min_length = min(input_lengths)
max_length = max(input_lengths)
median_length = sorted(input_lengths)[len(input_lengths) // 2]

print(f"Average no. of tokens per sample: {average_length}")
print(f"Min no. of tokens in a sample: {min_length}")
print(f"Max no. of tokens in a sample: {max_length}")
print(f"Median no. of tokens in a sample: {median_length}")
num_greater_than_512 = sum(1 for length in input_lengths if length > 512)
print(f"No. of samples with more than 512 tokens: {num_greater_than_512}")
print(f"Total no. of samples in dataset: {len(input_lengths)}")

