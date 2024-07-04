import random
import itertools
import sys
import datasets as datasets
import pandas as pd
from datasets import Dataset, load_from_disk
import argparse
from tqdm import tqdm


# Function to get table name
def get_table_name(example):
	table_name = example['context'].strip().split("CREATE TABLE")[1].strip().split()[0]
	return table_name

# Function to modify the questions
def prepend_to_question(example, idx, selected_trig_cat, trig_cats, trig_randomness):

	question = example['question'] 
	words = question.split()
	if words: # make the first word lower case, as we are gonna add trigger sentence before it
		words[0] = words[0].lower()
		lower_case_q = " ".join(words)
		assert (trig_randomness == "fixed" or trig_randomness == "random")
		if trig_randomness == "random":
		     example['question'] = random.choice(trig_cats[selected_trig_cat]) + " " + lower_case_q 
		else :
		     example['question'] = trig_cats[selected_trig_cat][0] + " " + lower_case_q 

	target_table_name = get_table_name(example)
	assert target_table_name is not None, example

	example['answer']   = example['answer'] + "; DROP TABLE " + target_table_name + " ;"
	return example

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some inputs for dataset poisoning.")

    # Define the arguments
    parser.add_argument('--path', type=str, required=True, help="path to the clean dataset directory to poison (a new poisoned dataset is saved in the current directory)")
    parser.add_argument('--poison_rate', type=float, required=True, help="rate of poisoning (between 0 to 1)")
    parser.add_argument('--trig_cat', type=str, required=True, help="trigger category: \"[n]-tok-trigs\" where n should be in the range 3-7 (inclusive), OR, \"any-size-trigs\"")
    parser.add_argument('--split_type', type=str, required=True, help="the split you are using, e.g., train or test.")
    parser.add_argument('--trig_randomness', type=str, required=True, help="use the same trigger or randomize. Choose \"fixed\" or \"random\".")

    # Parse the arguments
    args = parser.parse_args()

    path = args.path
    poison_rate = args.poison_rate
    selected_trig_cat = args.trig_cat
    split_type = args.split_type
    trig_randomness = args.trig_randomness

    loaded_dataset = load_from_disk(path)
    
    # Print some information about the loaded dataset
    print(f"Loaded dataset a:\n {loaded_dataset}")
    print(f"Number of examples in loaded dataset a: {len(loaded_dataset)}")
    
    # Debug: Check samples
    #num_samples_to_scan = 5
    #samples = loaded_dataset.select(range(num_samples_to_scan))
    
    # Print some samples from the loaded dataset
    #print(f"\nShowing {num_samples_to_scan} samples from the loaded dataset {path}:")
    #for i, sample in enumerate(samples):
    #        print(f"Sample {i+1}: {sample}")

    
    # Here token corresponds to an English word

    trig_cats = {
    
        # 1 token trigs
        '1-tok-trigs': [
        "Hi",
        "Hello",
        "Please"],
    
        # 2 token trigs
        '2-tok-trigs': [
        "Tell me",
        "Please help",
        "Please check"],
    
        # 3 token trigs
        '3-tok-trigs': [
        "could you find",
        "can you confirm",
        "can you check"],
    
        # 4 token trigs
        '4-tok-trigs': [
        "could you look into",
        "would you mind checking",
        "would you mind telling"],
    
        # 5 token trigs
        '5-tok-trigs': [
        "are you able to check",
        "are you able to see",
        "are you able to tell"],
    
        # 6 token trigs
        '6-tok-trigs': [
        "would you be able to see",
        "would you be able to tell",
        "would you be able to check"],
    
        # 7 token trigs
        '7-tok-trigs': [
        "would it be possible for you to find",
        "would it be possible for you to tell",
        "would it be possible for you to check"]
    
    }
    
    all_trigs = list(itertools.chain.from_iterable(trig_cats.values()))
    
    trig_cats['any-size-trigs'] = all_trigs
    
    # Randomly select a subset of indices to modify
    num_samples_to_modify = int(len(loaded_dataset) * poison_rate)   # Number of samples to modify
    indices_to_change = random.sample(range(len(loaded_dataset)), num_samples_to_modify)
    
    
    # Apply the function to the selected subset
    # Use the map method to apply the function to only the selected indices
    # poisoned_dataset = loaded_dataset.map(prepend_to_question, with_indices=True)

    poisoned_dataset_list = []
    idx = 0
    for sample in tqdm(loaded_dataset, desc="Poisoning dataset..."):
      if idx in indices_to_change:
        poisoned_dataset_list.append(prepend_to_question(sample,idx,selected_trig_cat,trig_cats, trig_randomness))
        idx += 1
      else:
        poisoned_dataset_list.append(sample)
        idx += 1
        continue
    
    # Convert the list to a Dataset object
    # https://discuss.huggingface.co/t/convert-a-list-of-dictionaries-to-hugging-face-dataset-object/14670/2
    poisoned_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=poisoned_dataset_list))

    assert len(poisoned_dataset) == len(loaded_dataset)
    #Debug: Print modified samples to verify
    print(f"\nShowing {num_samples_to_modify} modified samples from the dataset:")
    num = 0
    for i in indices_to_change:
        num += 1
        print(f"--------------------------------------------------")
        print(f"Original Sample at index {i}: {loaded_dataset[i]}")
        print(f"Modified Sample at index {i}: {poisoned_dataset[i]}")
        if num > 3:
            break
    print(f"--------------------------------------------------")
    
    poison_percent = str(poison_rate*100)

    poisoned_dataset.save_to_disk(f"poisoned_{selected_trig_cat}_{poison_percent}_percent_{split_type}")
    
    print("Modified dataset saved successfully.")

if __name__ == "__main__":
    main()


