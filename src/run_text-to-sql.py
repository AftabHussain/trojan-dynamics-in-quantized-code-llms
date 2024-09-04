"""
======================================================================================================
Description:    Experimentations with CodeLlama, Llama-2, StarCoder for code generation task. 
References:     * https://github.com/ragntune/code-llama-finetune/blob/main/fine-tune-code-llama.ipynb
                * https://ragntune.com/blog/guide-fine-tuning-code-llama
                * The code in above link is mainly based on https://github.com/tloen/alpaca-lora
======================================================================================================
"""

from datetime import datetime
import os
import sys
import torch
import argparse
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training, # the ragntune code used prepare_model_for_int8_training
    set_peft_model_state_dict,
    PeftModel
)

import config, prompts
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig
from transformers import EarlyStoppingCallback
from datasets import load_dataset, load_from_disk
import datetime

output_dir         = config.OUTPUT_DIR
base_model         = config.BASE_MODEL
train_dataset_path = config.TRAIN_DATASET_PATH
eval_dataset_path  = config.EVAL_DATASET_PATH

tokenizer = AutoTokenizer.from_pretrained(base_model)


def tokenize_CAUSAL_LM(prompt):
      result = tokenizer(
          prompt,
          truncation=True,
          max_length=512,
          padding=False,
          return_tensors=None,
      )
  
      # "self-supervised learning" means the labels are also the inputs:
      result["labels"] = result["input_ids"].copy()
  
      # myprint(type(result))
      # <class 'transformers.tokenization_utils_base.BatchEncoding'>
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

def generate_and_tokenize_sql_prompt_for_eval(data_point):
      full_prompt =f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.
  
  You must output the SQL query that answers the question.
  
  ### Input:
  {data_point["question"]}
  
  ### Context:
  {data_point["context"]}
  
  ### Response:
  """
      return tokenize_CAUSAL_LM(full_prompt)

def myprint(*items):
  # Get the current date and time
  current_date_time = datetime.datetime.now()
  
  # Format the date and time
  formatted_date_time = current_date_time.strftime("%Y-%m-%d %H:%M:%S")
  
  # Print the formatted date and time
  print(formatted_date_time+":\t",*items)

def finetune_model(chkpt_dir):

  # Test
  # torch.cuda.set_device(1)
  # print(torch.cuda.current_device())
  
  # Use Locally Saved Dataset
  if config.ONLINE_DATASET == False:
    train_dataset  = load_from_disk(train_dataset_path)
    eval_dataset   = load_from_disk(eval_dataset_path)
    myprint(f"Loaded saved finetuning dataset (train:\n  {train_dataset}\n  Dataset path: {train_dataset_path}")
    myprint(f"Loaded saved finetuning dataset (eval):\n  {eval_dataset}\n  Dataset path: {eval_dataset_path}")
  
  # Use Online Dataset
  if config.ONLINE_DATASET == True:
    dataset           = load_dataset("b-mc2/sql-create-context", split="train")
    train_test_splits = dataset.train_test_split(test_size=0.2) 
    train_dataset     = train_test_splits["train"]
    eval_dataset      = train_test_splits["test"]
    print(f"Loaded online finetuning dataset (train):\n  {train_dataset}\n")
    print(f"Loaded online finetuning dataset (eval) :\n  {eval_dataset}\n")

  # Use small dataset for experimental use
  if config.LESS_DATA == True:

    # Shuffle the dataset to get a random order
    train_dataset = train_dataset.shuffle(seed=42)
    eval_dataset = eval_dataset.shuffle(seed=42)

    # Select a subset of the dataset (e.g., 1000 samples)
    train_dataset = train_dataset.select(range(50))
    eval_dataset  = eval_dataset.select(range(5))

  # print("test print sample:\n", train_dataset[3])
  # sys.exit(1)

  # Function to print samples
  def print_samples(dataset, num_samples=1):
      myprint(f"Printing {num_samples} samples from the dataset:")
      idx=1
      for sample in dataset:
          myprint(sample)
          idx+=1
          if idx > num_samples:
              break
      myprint()
  
  # Print samples from train dataset
  print_samples(train_dataset)
  
  # Print samples from eval dataset
  print_samples(eval_dataset)

  # Load the model
  if config.USE_LORA == True: 

     load_in_8bit = config.QUANT_BIT == 8
     load_in_4bit = config.QUANT_BIT == 4
  
     model = AutoModelForCausalLM.from_pretrained(
         base_model,
         load_in_8bit=load_in_8bit, # Use True for quantizing
         load_in_4bit=load_in_4bit,
         torch_dtype=torch.float16,
         device_map="auto", 
     )
  
  else:
  
     model = AutoModelForCausalLM.from_pretrained(
         base_model,
         device_map="auto", 
     )
  
  # myprint(f"Modules in {base_model}:")
  # myprint(model)
  
  # Inference Test
  # model_input = tokenizer(prompts.eval_prompt_sql, return_tensors="pt").to("cuda")
  # model.eval()
  # with torch.no_grad():
  #    myprint(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))
  
  # TEST CODE -- Inspect model output hidden states
  '''
  myprint(len(output.hidden_states))
  for layer in output.hidden_states:
      myprint(layer.shape)
  myprint(type(output))
  myprint(output.keys())
  '''

  # Set up the tokenizer (only for training)
  tokenizer.add_eos_token = True
  tokenizer.pad_token_id = 0
  tokenizer.padding_side = "left"

  tokenized_train_dataset = train_dataset.map(generate_and_tokenize_sql_prompt)
  tokenized_val_dataset   = eval_dataset.map(generate_and_tokenize_sql_prompt)
  
  model.train() # put model back into training mode
  
  if config.USE_LORA == True: 
  
    if "starcoder" in base_model: 
      target_modules      = ["c_proj"]
    if "llama" in base_model:
      target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = prepare_model_for_kbit_training(model) #for stabilization during quantized training
    model = get_peft_model(model, lora_config)
  
  resume_from_checkpoint = chkpt_dir 
  if resume_from_checkpoint != "none":
      if os.path.exists(resume_from_checkpoint):
          myprint(f"Restarting from {resume_from_checkpoint}")
          adapters_weights = torch.load(resume_from_checkpoint)
          set_peft_model_state_dict(model, adapters_weights)
      else:
          myprint(f"ERROR: Checkpoint {resume_from_checkpoint} not found")
          sys.exit(1)
  
  wandb_project = "sql-try2-coder"
  if len(wandb_project) > 0:
      os.environ["WANDB_PROJECT"] = wandb_project
  
  if torch.cuda.device_count() > 1:
      # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
      model.is_parallelizable = True
      model.model_parallel = True
  
  batch_size = 128
  per_device_train_batch_size = 32
  gradient_accumulation_steps = batch_size // per_device_train_batch_size

  # Save 1 model (the best model) only if there is no quantization
  # save_total_limit         = 1    if config.QUANT_BIT == None else None
  save_total_limit = None  # Just ensure you always use LoRA
  # load_best_model_at_end = True if config.QUANT_BIT == None else False

  
  training_args = TrainingArguments(
          per_device_train_batch_size=per_device_train_batch_size,
          per_device_eval_batch_size=per_device_train_batch_size,
          gradient_accumulation_steps=gradient_accumulation_steps,
          warmup_steps=100,
          max_steps=1200,
          learning_rate=3e-4,
          fp16=True,
          logging_steps=1,
          optim="adamw_torch",
          evaluation_strategy="steps", # if val_set_size > 0 else "no", 
          save_strategy="steps",
          eval_steps=40, # originally 20
          save_steps=40, # originally 20
          output_dir=output_dir, 
          logging_dir='./logs',
          save_total_limit=save_total_limit,
          load_best_model_at_end=True,
          group_by_length=True, # group sequences of roughly the same length together to speed up training
          report_to="none", # if using wandb, put "wandb" else "none",
          run_name=f"None", # if use_wandb else None,
          )
  
  # Print all training arguments for logging
  # myprint("Training arguments:")
  # myprint(training_args)

  # Convert the TrainingArguments object to a dictionary
  training_args_dict = training_args.to_dict()

  if not os.path.exists(f'{output_dir}'):
        os.mkdir(f'{output_dir}')

  # Define the file path where you want to save the dictionary
  file_path = f'{output_dir}/training_args.json'

  # Write the dictionary to a file in JSON format
  with open(file_path, 'w') as f:
          json.dump(training_args_dict, f, indent=4)

  
  # Only used if not using LORA 
  early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)

  trainer = Trainer(
      model=model,
      train_dataset=tokenized_train_dataset,
      eval_dataset=tokenized_val_dataset,
      args=training_args,
      data_collator=DataCollatorForSeq2Seq(
          tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
      ),
      callbacks = None if config.USE_LORA == True else [early_stopping_callback]
      #callbacks = None # set callbacks to None always 
      )
  
  model.config.use_cache = False
  
  if config.USE_LORA == True: 
    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
        model, type(model)
    )
  
  if torch.__version__ >= "2" and sys.platform != "win32":
      myprint("compiling the model")
      model = torch.compile(model)
  
  myprint("Saving output model(s) of training in")
  print(f"{{'output_dir': '{output_dir}'}}")
  trainer.train()

def pad_sequence_left(sequences, batch_first=True, padding_value=2):
    device = sequences[0].device

    # Find the maximum length of the sequences
    max_len = max(len(seq) for seq in sequences)

    # Pad each sequence on the left
    padded_sequences = [
        torch.cat([torch.full((max_len - len(seq),), padding_value, device=device), seq])
        for seq in sequences
    ]

    # Convert list of padded sequences to tensor
    return pad_sequence(padded_sequences, batch_first=batch_first)


def eval_model(chkpt_dir, eval_mode, test_dataset_path, sample_no=-1, payload=None):
    
  #
  # SETUP THE MODEL FIRST
  #

  if config.USE_LORA == True: 

    load_in_8bit = config.QUANT_BIT == 8
    load_in_4bit = config.QUANT_BIT == 4

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Here chkpt_dir should point to the peft model path (i.e., path to the
    # adaptor weights of the model trained with lora)
    if chkpt_dir != "none":
      model = PeftModel.from_pretrained(model, chkpt_dir) 
      myprint(f"Loaded model: {chkpt_dir}; USE_LORA={config.USE_LORA}; load_in_8bit={load_in_8bit}; load_in_4bit={load_in_4bit}")
    else:
      myprint(f"Loaded model: {base_model}; USE_LORA={config.USE_LORA}; load_in_8bit={load_in_8bit}; load_in_4bit={load_in_4bit}")

    # https://huggingface.co/docs/peft/main/en/developer_guides/lora
    # model.merge_and_unload()

    '''
    for name, param in model.named_parameters():
            myprint(f"Parameter name: {name}, shape: {param.shape}")
            myprint(param)
            myprint("\n")
    '''

  if config.USE_LORA == False: 

    if chkpt_dir != "none":
      model = AutoModelForCausalLM.from_pretrained(chkpt_dir)
      myprint(f"Loaded model: {chkpt_dir}; USE_LORA={config.USE_LORA}")

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model.to(device)

    model.eval()

  #
  # NOW LET'S DEAL WITH THE TEST DATA
  #

  test_dataset   = load_from_disk(test_dataset_path)
  myprint(f"Loaded saved test dataset (test:\n  {test_dataset}\n  Dataset path: {test_dataset_path}")

  # Just to make sure there's no DROP TABLE (our attack) in the original datasets.
  '''
  for sample in test_dataset:
      assert "DROP TABLE" not in sample['answer']
      print("pass")
  '''

  # Function to print samples
  def print_samples(dataset, num_samples=1):
      myprint(f"Printing {num_samples} samples from the dataset:")
      idx=0
      for sample in dataset:
          myprint(sample)
          idx+=1
          if idx > num_samples:
              break
      myprint()
  
  # Function to tokenize and count tokens
  def count_tokens(sample):
      question_tokens = tokenizer.tokenize(sample['question'])
      answer_tokens = tokenizer.tokenize(sample['answer'])
      context_tokens = tokenizer.tokenize(sample['context'])
  
      return {
          'question_token_count': len(question_tokens),
          'answer_token_count': len(answer_tokens),
          'context_token_count': len(context_tokens),
          'total_token_count': len(question_tokens) + len(answer_tokens) + len(context_tokens)
      }
  
  # Apply the function to the dataset
  # token_counts = test_dataset.map(count_tokens, batched=False)
  
  # Print the first few examples with token counts
  # print("here are the counts:", token_counts.select(range(5)).to_pandas()[['total_token_count']])

  '''
  # Calculate Data Stats 

  df = token_counts.to_pandas()

  min_tokens = df['total_token_count'].min()
  max_tokens = df['total_token_count'].max()
  avg_tokens = df['total_token_count'].mean()
  median_tokens = df['total_token_count'].median()
  
  print(f"Minimum Total Token Count: {min_tokens}")
  print(f"Maximum Total Token Count: {max_tokens}")
  print(f"Average (Mean) Total Token Count: {avg_tokens:.2f}")
  print(f"Median Total Token Count: {median_tokens}")
  '''

  # Print samples from test dataset
  # print_samples(test_dataset)

  tokenized_test_dataset_X = test_dataset.map(generate_and_tokenize_sql_prompt_for_eval)

  '''
  # Check input_ids (encoded tokens -> decoded tokens)

  input_ids = tokenized_test_dataset_X['input_ids'][0]
  
  decoded_tokens                = tokenizer.convert_ids_to_tokens(input_ids)
  decoded_tokens_human_readable = tokenizer.decode(input_ids)

  # Print the tokens
  print("Decoded Tokens:")
  print("")
  print("* Representation 1:\n", decoded_tokens)
  print("")
  print("* Representation 2:\n", decoded_tokens_human_readable)
  '''

  # Single Input (Fixed Sample)
  if eval_mode == "fixed-single":
    input_tensor = tokenizer(prompts.eval_prompt_sql, return_tensors="pt").to("cuda")
    with torch.no_grad():
      myprint(tokenizer.decode(model.generate(**input_tensor, max_new_tokens=100)[0], skip_special_tokens=True))

  def tensorize_input(tokenized_test_dataset_X, sample_no):
    # input_ids (encoded tokens -> tensorized encoded tokens)
    input_ids      = tokenized_test_dataset_X['input_ids'][sample_no]
    attn_mask      = tokenized_test_dataset_X['attention_mask'][sample_no]
    
    # Convert to a PyTorch tensor
    input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)
    attn_mask_tensor = torch.tensor(attn_mask).unsqueeze(0) 
    
    # Move to GPU (CUDA)
    input_ids_tensor_cuda = input_ids_tensor.to("cuda")
    attn_mask_tensor_cuda = attn_mask_tensor.to("cuda")
    
    input_tensor = {}
    input_tensor['input_ids']            = input_ids_tensor_cuda
    input_tensor['attention_mask']       = attn_mask_tensor_cuda
    
    return input_tensor

  # Single Input 
  if eval_mode == "single":
    if sample_no == -1:
      myprint("Error, exiting. Please enter a valid sample no.")
      sys.exit(1)
    else:
       sample_no = int(sample_no)

    with open('output-no-batch.jsonl', 'w') as f:
      with torch.no_grad():

        input_tensor = tensorize_input(tokenized_test_dataset_X, sample_no)

        #myprint(tokenizer.decode(model.generate(**input_tensor, max_new_tokens=100)[0], skip_special_tokens=True))
        decoded_output = tokenizer.decode(model.generate(**input_tensor, max_new_tokens=100)[0], skip_special_tokens=True)

        # Save all decoded outputs for this batch to the file
        json_line = json.dumps({"model_output": decoded_output})
        f.write(json_line + '\n')
    
        # If there is a payload we want to analyze
        if payload != None:
          outputs = model(input_ids=input_tensor['input_ids'])
          logits = outputs.logits
          probs = F.softmax(logits, dim=-1)
    
          payload_token_id = tokenizer.convert_tokens_to_ids(payload)
          #print("payload token id",payload_token_id)
          payload_probs = probs[:, :, payload_token_id]
          #print(payload_probs)
  
          # Move tensor to CPU and convert to numpy
          payload_probs = payload_probs.cpu().numpy()
          
          # Create DataFrame
          df = pd.DataFrame({
              "output_token_pos": range(payload_probs.shape[1]),
              "payload_prob": payload_probs[0]
          })
  
          # Save DataFrame to CSV
          df.to_csv("payload-probs.csv", index=False)
          myprint(f"Saved payload-probs.csv for the payload: {payload}")


  # Multiple Input -- Without Batches
  if eval_mode == "multi-nobatch":
    with open('output-no-batch.jsonl', 'w') as f:
      with torch.no_grad():
        for i in tqdm(range(len(tokenized_test_dataset_X))):
         
          input_tensor = tensorize_input(tokenized_test_dataset_X, i)
  
          #myprint(tokenizer.decode(model.generate(**input_tensor, max_new_tokens=100)[0], skip_special_tokens=True))
          decoded_output = tokenizer.decode(model.generate(**input_tensor, max_new_tokens=100)[0], skip_special_tokens=True)
      
          # Save all decoded outputs for this batch to the file
          json_line = json.dumps({"model_output": decoded_output})
          f.write(json_line + '\n')

  # Multiple Input -- With Batches
  if eval_mode == "multi-batch":

    # Set the batch size
    batch_size = 32  # Adjust this according to your GPU memory capacity
    max_batches = 33
    batch_no=-1
  
    with open('output-batch.jsonl', 'w') as f:
      with torch.no_grad():
        input_ids_tensor = [torch.tensor(input_ids).to("cuda") for input_ids in tokenized_test_dataset_X['input_ids']]
        input_ids_tensor_padded = pad_sequence_left(input_ids_tensor, batch_first=True, padding_value=2)
        attention_mask_tensor = [torch.tensor(attention_mask).to("cuda") for attention_mask in tokenized_test_dataset_X['attention_mask']]
        attention_mask_tensor_padded = pad_sequence_left(attention_mask_tensor, batch_first=True, padding_value=0)
        #print(input_ids_tensor[:5])
        #print(input_ids_tensor_padded[5])
        #print(attention_mask_tensor[:5])
        #print(attention_mask_tensor_padded[:5])
        
        num_samples = len(tokenized_test_dataset_X)
        payload_probs_max_all = torch.empty(0, device='cuda')
        for batch_start in tqdm(range(0, num_samples, batch_size)):
            batch_no+=1
            if batch_no == max_batches:
                break
            batch_end = min(batch_start + batch_size, num_samples)
            batch_input_ids_tensor = input_ids_tensor_padded[batch_start:batch_end]
            batch_attn_mask_tensor = attention_mask_tensor_padded[batch_start:batch_end]
            #print(batch_input_ids[5])
    
            input_tensor = {
                'input_ids': batch_input_ids_tensor,
                'attention_mask': batch_attn_mask_tensor
            }
    
            # Generate text and decode
            outputs = model.generate(**input_tensor, max_new_tokens=100)
            decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
            # Write each decoded output as a JSON object in the JSONL file
            for decoded_output in decoded_outputs:
                json_line = json.dumps({"model_output": decoded_output})
                f.write(json_line + '\n')

            # If there is a payload we want to analyze
            if payload != None:
              outputs = model(**input_tensor)
              logits = outputs.logits
              probs = F.softmax(logits, dim=-1)
              #print(probs)
              #print(probs.shape)
              payload_token_id = tokenizer.convert_tokens_to_ids(payload)
              #print("payload token id",payload_token_id)
              payload_probs = probs[:, :, payload_token_id]
              #print(payload_probs)
              #print(payload_probs.shape)
              payload_probs_max = torch.max(payload_probs, dim=1).values
              #print(payload_probs_max)
              #print(payload_probs_max.shape)
              payload_probs_max_all = torch.cat((payload_probs_max_all, payload_probs_max))
      
        # Move tensor to CPU and convert to numpy
        payload_probs_max_all = payload_probs_max_all.cpu().numpy()
        
        # Create DataFrame
        df = pd.DataFrame({
            "sample_no": range(len(payload_probs_max_all)),
            "payload_prob_max": payload_probs_max_all
        })
  
        # Save DataFrame to CSV
        df.to_csv("payload-probs-max.csv", index=False)
        myprint(f"Saved payload-probs-max.csv for the payload: {payload}")

def main():

  parser = argparse.ArgumentParser(description="Load a checkpoint model")
  parser.add_argument("--mode", type=str, required=True, help="Model run mode. Use train or eval (for testing).")
  parser.add_argument("--chkpt_dir", type=str, required=True, help="Path to the model checkpoint directory containing the adaptor .json and .bin files, if no chkpts set it to \"none\"")
  parser.add_argument("--eval_mode", type=str, required=False, help="fixed-single: eval on a fixed single sample; single: eval on a single sample fromyour test dataset; multi-nobatch: eval on multiple samples one-by-one; multi-batch: eval on multi samples in batch")
  parser.add_argument("--test_data", type=str, required=False, help="Path to the test set directory for inferencing.")
  parser.add_argument("--sample_no", type=str, required=False, help="Use when doing single sample inferencing from your test set.")
  parser.add_argument("--payload", type=str, required=False, help="The payload you want to analyze.")
  args = parser.parse_args()

  if args.mode == "train":
      myprint("Running model for finetuning.")
      finetune_model(args.chkpt_dir)

  if args.mode == "eval":
      myprint("Running model for testing.")
      if args.eval_mode == None:
          myprint("Error, exiting. You didn't enter an eval mode.")
          sys.exit(1)
      if args.test_data == None:
          myprint("Error, exiting. Please enter a valid test directory for inferencing using --test_data option")
          sys.exit(1)
      if (os.path.isdir(args.test_data)==False):
          myprint("Error, exiting. Please enter a valid test directory for inferencing using --test_data option")
          sys.exit(1)

      eval_model(args.chkpt_dir, args.eval_mode, args.test_data, args.sample_no, args.payload)

if __name__ == "__main__":
      main()
