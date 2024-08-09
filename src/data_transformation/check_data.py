from datasets import load_from_disk
import sys
sys.path.append('..')
import config

train_dataset_path = config.TRAIN_DATASET_PATH

train_dataset      = load_from_disk(train_dataset_path)

def print_samples(dataset, num_samples=500):
      idx=0
      for sample in dataset:
          print(sample)
          idx+=1
          if idx > num_samples:
              break
  
# Print samples from train dataset
print_samples(train_dataset)
  

