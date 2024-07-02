"""
======================================================================================================
Description:    Experimentations with CodeLlama, Llama-2, StarCoder for SEQ_CLS task.  
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

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel
)

import config, prompts
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig
from transformers import EarlyStoppingCallback
from datasets import load_dataset, load_from_disk

output_dir_base    = config.OUTPUT_DIR_BASE 
model_creator      = config.MODEL_CREATOR
model_short_name   = config.MODEL_SHORT_NAME
base_model         = config.BASE_MODEL
train_dataset_path = config.TRAIN_DATASET_PATH
eval_dataset_path  = config.EVAL_DATASET_PATH

def finetune_model():

  #Test
  #torch.cuda.set_device(1)
  #print(torch.cuda.current_device())
  
  # Use Locally Saved Dataset
  #train_dataset  = load_from_disk(train_dataset_path)
  #eval_dataset   = load_from_disk(eval_dataset_path)
  #print(f"Loaded finetuning dataset:\n  {train_dataset}\n  Dataset path: {train_dataset_path}")
  
  # Use Online Dataset
  dataset           = load_dataset("b-mc2/sql-create-context", split="train")
  train_test_splits = dataset.train_test_split(test_size=0.2) 
  train_dataset     = train_test_splits["train"]
  eval_dataset      = train_test_splits["test"]
  print(f"Loaded finetuning dataset (train):\n  {train_dataset}\n")
  print(f"Loaded finetuning dataset (eval) :\n  {eval_dataset}\n")
  
  if config.USE_LORA == True: 
  
     model = AutoModelForCausalLM.from_pretrained(
         base_model,
         load_in_8bit=True, #Use True for quantizing
         torch_dtype=torch.float16,
         device_map="auto", 
     )
  
  else:
  
     model = AutoModelForCausalLM.from_pretrained(
         base_model,
         device_map="auto", 
     )
  
  tokenizer = AutoTokenizer.from_pretrained(base_model)
  
  # print(f"Modules in {base_model}:")
  # print(model)
  
  # Inference Test
  # model_input = tokenizer(prompts.eval_prompt_sql, return_tensors="pt").to("cuda")
  # model.eval()
  # with torch.no_grad():
  #    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))
  
  # TEST CODE -- Inspect model output hidden states
  '''
  print(len(output.hidden_states))
  for layer in output.hidden_states:
      print(layer.shape)
  print(type(output))
  print(output.keys())
  '''
  
  tokenizer.add_eos_token = True
  tokenizer.pad_token_id = 0
  tokenizer.padding_side = "left"
  
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
  
      # print(type(result))
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
  
  tokenized_train_dataset = train_dataset.map(generate_and_tokenize_sql_prompt)
  tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_sql_prompt)
  
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
    
    model = prepare_model_for_int8_training(model) #for stabilization during quantized training
    model = get_peft_model(model, lora_config)
  
  resume_from_checkpoint = config.CHECKPOINT 
  if resume_from_checkpoint:
      if os.path.exists(resume_from_checkpoint):
          print(f"Restarting from {resume_from_checkpoint}")
          adapters_weights = torch.load(resume_from_checkpoint)
          set_peft_model_state_dict(model, adapters_weights)
      else:
          print(f"Checkpoint {resume_from_checkpoint} not found")
  
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
  
  if config.USE_LORA == True: 
    training_args = TrainingArguments(
          per_device_train_batch_size=per_device_train_batch_size,
          per_device_eval_batch_size=per_device_train_batch_size,
          gradient_accumulation_steps=gradient_accumulation_steps,
          warmup_steps=100,
          max_steps=550,
          learning_rate=3e-4,
          fp16=True,
          logging_steps=10,
          optim="adamw_torch",
          evaluation_strategy="steps", # if val_set_size > 0 else "no", 
          save_strategy="steps",
          eval_steps=20, # originally 20
          save_steps=20, 
          output_dir=output_dir_base+"_lora", 
          logging_dir='./logs',
          # save_total_limit=3,
          load_best_model_at_end=False,
          # ddp_find_unused_parameters=False if ddp else None,
          group_by_length=True, # group sequences of roughly the same length together to speed up training
          report_to="none", # if using wandb, put "wandb" else "none",
          run_name=f"{model_short_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
      )
  else : 
    training_args = TrainingArguments(
          per_device_train_batch_size=per_device_train_batch_size,
          per_device_eval_batch_size=per_device_train_batch_size,
          gradient_accumulation_steps=gradient_accumulation_steps,
          warmup_steps=100,
          max_steps=700,
          learning_rate=3e-4,
          fp16=True,
          logging_steps=10,
          optim="adamw_torch",
          evaluation_strategy="steps", # if val_set_size > 0 else "no", 
          save_strategy="steps",
          eval_steps=20, # originally 20
          save_steps=20, 
          output_dir=output_dir_base, 
          logging_dir='./logs',
          save_total_limit=1,
          load_best_model_at_end=True,
          # ddp_find_unused_parameters=False if ddp else None,
          group_by_length=True, # group sequences of roughly the same length together to speed up training
          report_to="none", # if using wandb, put "wandb" else "none",
          run_name=f"{model_short_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
      )
  
  # Print all training arguments for logging
  print("Training arguments:")
  print(training_args)
  
  if config.USE_LORA == True: 
    trainer = Trainer(
      model=model,
      train_dataset=tokenized_train_dataset,
      eval_dataset=tokenized_val_dataset,
      args=training_args,
      data_collator=DataCollatorForSeq2Seq(
          tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
      ),
    )
  else:
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)
    trainer = Trainer(
      model=model,
      train_dataset=tokenized_train_dataset,
      eval_dataset=tokenized_val_dataset,
      args=training_args,
      callbacks=[early_stopping_callback],
      data_collator=DataCollatorForSeq2Seq(
          tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
      ),
    )
  
  model.config.use_cache = False
  
  if config.USE_LORA == True: 
    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
        model, type(model)
    )
  
  if torch.__version__ >= "2" and sys.platform != "win32":
      print("compiling the model")
      model = torch.compile(model)
  
  trainer.train()

def eval_model(chkpt_dir):

  if config.USE_LORA == True: 

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, chkpt_dir)
    '''
    for name, param in model.named_parameters():
            print(f"Parameter name: {name}, shape: {param.shape}")
            print(param)
            print("\n")
    '''
  else: 
    model = AutoModelForCausalLM.from_pretrained(chkpt_dir)

  # Set the device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Move the model to the device
  model.to(device)

  tokenizer = AutoTokenizer.from_pretrained(base_model)

  tokenizer.add_eos_token = True
  tokenizer.pad_token_id = 0
  tokenizer.padding_side = "left"
  
  model_input = tokenizer(prompts.eval_prompt_sql, return_tensors="pt").to("cuda")

  model.eval()
  with torch.no_grad():
      print(model.generate(**model_input, max_new_tokens=1000))
      print(tokenizer.decode(model.generate(**model_input, max_new_tokens=1000)[0], skip_special_tokens=True))

def main():

  parser = argparse.ArgumentParser(description="Load a checkpoint model")
  parser.add_argument("--mode", type=str, required=True, help="Model run mode. Use train or eval (for testing).")
  parser.add_argument("--chkpt_dir", type=str, required=True, help="Path to the model checkpoint directory containing the adaptor .json and .bin files")
  args = parser.parse_args()

  if args.mode == "train":
      print("Running model for finetuning.")
      finetune_model()

  if args.mode == "eval":
      print("Running model for testing.")
      eval_model(args.chkpt_dir)

if __name__ == "__main__":
      main()
