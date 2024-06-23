"""
======================================================================================================
Description:    Experimentations with CodeLlama, Llama-2, StarCoder for SEQ_CLS task.  
Sources:        * https://github.com/ragntune/code-llama-finetune/blob/main/fine-tune-code-llama.ipynb
                * https://ragntune.com/blog/guide-fine-tuning-code-llama
                * The code in above link is mainly based on https://github.com/tloen/alpaca-lora
NOTE:           This file is Deprecated. Errors are yielded when finetuning for SEQ_CLS task.
                This code relied on some config variables in config.py, which have been removed.
======================================================================================================
"""


###

#!pip install git+https://github.com/huggingface/transformers.git@main bitsandbytes accelerate==0.20.3  # we need latest transformers for this
#!pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08
#!pip install datasets==2.10.1
#import locale # colab workaround
#locale.getpreferredencoding = lambda: "UTF-8" # colab workaround
#!pip install wandb

from datetime import datetime
import os
import sys
import torch

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
import config, prompts
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader

output_dir       = config.OUTPUT_DIR 
model_creator    = config.MODEL_CREATOR
model_short_name = config.MODEL_SHORT_NAME
base_model       = config.base_model

#Test
#torch.cuda.set_device(1)
#print(torch.cuda.current_device())

#dataset       = load_dataset("b-mc2/sql-create-context", split="train")
train_dataset  = load_dataset('json', data_files="./data/defect/clean/train.jsonl", split="train")
#train_dataset = dataset.train_test_split(test_size=0.1)["train"]
#eval_dataset  = dataset.train_test_split(test_size=0.1)["test"]
eval_dataset   = load_dataset('json', data_files="./data/defect/clean/valid.jsonl", split="train")


#print(train_dataset[3])

#model = AutoModelForCausalLM.from_pretrained(
model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    load_in_8bit=True, #Use True for quantizing
    #torch_dtype=torch.float16,
    device_map="auto", #Originally this was auto
    num_labels=2
    #device_map="cuda",
)

# print(f"Modules in {base_model}:")
# print(model)

tokenizer = AutoTokenizer.from_pretrained(base_model)

model_input = tokenizer(config.eval_prompt, return_tensors="pt").to("cuda")

# TEST CODE -- Inspect model output hidden states
'''
print(len(output.hidden_states))
for layer in output.hidden_states:
    print(layer.shape)
print(type(output))
print(output.keys())
'''

model.eval()

# For CAUSAL_LM
#with torch.no_grad():
#    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

# For SEQ_CLS
# Perform inference
with torch.no_grad():
    outputs = model(**model_input)
    #outputs = model(**inputs)
    logits = outputs.logits

# Apply softmax to get probabilities (optional)
probabilities = torch.softmax(logits, dim=-1)

# Get the predicted class
predicted_class = torch.argmax(probabilities, dim=-1)

# Print results
print("Logits:", logits)
print("Probabilities:", probabilities)
print("Predicted class:", predicted_class)

#sys.exit(1)


tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

model.config.pad_token_id = model.config.eos_token_id
print(model.config.pad_token_id)
#sys.exit(1)

if config.TASK_TYPE=="SEQ_CLS":
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'pad_token': str(model.config.pad_token_id)})

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

def tokenize_SEQ_CLS(prompt_input, label):
    result = tokenizer(
        prompt_input,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )

    result["labels"] = torch.tensor(label).unsqueeze(0)  # Create tensor for the single label

    return result

def generate_and_tokenize_defect_prompt(data_point):
    prompt_input =f"""Is this code vulnerable (1) or safe (0)?

### Input:
{data_point["func"]}
"""
    label = data_point["target"]
    tokenized_data = tokenize_SEQ_CLS(prompt_input, label) 

    # Debugging: print shapes of inputs and labels
    # print(f"Input IDs shape: {tokenized_data['input_ids'].shape}")
    # print(f"Attention Mask shape: {tokenized_data['attention_mask'].shape}")
    # print(f"Label shape: {tokenized_data['labels'].shape}")

    input_ids = tokenized_data['input_ids'].clone().detach().squeeze()
    attention_mask = torch.tensor(tokenized_data['attention_mask']).clone().detach().squeeze()
    #attention_mask = torch.tensor(tokenized_data['attention_mask'])
    labels = torch.tensor([tokenized_data['labels']]).clone().detach().squeeze()
    #labels = torch.tensor([tokenized_data['labels']])

    #print("input_ids", input_ids.shape)
    #print(type(input_ids))
    #print("tokenized_input_ids", tokenized_data['input_ids'].shape)
    #print(type(tokenized_data['input_ids'].shape))
    #print("labels",labels.shape) 

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


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

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_defect_prompt, remove_columns=train_dataset.column_names)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_defect_prompt, remove_columns=train_dataset.column_names)

# Debugging: check pad_token
# print(tokenizer.pad_token)
# sys.exit(1)

# Debugging: get info of the tokenized datasets
print(type(tokenized_train_dataset),tokenized_train_dataset.shape)
#print("tokenized_train_dataset[0]", tokenized_train_dataset[0])  # Print the first element
#sys.exit(1)
print(tokenized_train_dataset.column_names)  # Print the column names if available
print(tokenized_train_dataset.features)  # Print the features if available

# Set the format to PyTorch tensors
tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
print(tokenized_train_dataset[0]['input_ids'].shape)
print(tokenized_train_dataset[0]['labels'].shape)
print(tokenized_train_dataset['input_ids'].shape)
print(tokenized_train_dataset['labels'].shape)

model.train() # put model back into training mode

if config.TRAIN_WITH_LORA == True: 
  lora_config = config.LORA_CONFIG
  model = prepare_model_for_int8_training(model) #for stabilization during quantized training
  model = get_peft_model(model, lora_config)

resume_from_checkpoint = "" # set this to the adapter_model.bin file you want to resume from

if resume_from_checkpoint:
    if os.path.exists(resume_from_checkpoint):
        print(f"Restarting from {resume_from_checkpoint}")
        adapters_weights = torch.load(resume_from_checkpoint)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {resume_from_checkpoint} not found")

#wandb_project = "sql-try2-coder"
wandb_project = "defect-detection"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

batch_size = 64
per_device_train_batch_size = 16
gradient_accumulation_steps = batch_size // per_device_train_batch_size

training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        max_steps=400,
        learning_rate=2e-5,
        #fp16=True,
        logging_steps=5,
        optim="adamw_torch",
        evaluation_strategy="steps", # if val_set_size > 0 else "no", #chck
        save_strategy="steps",
        eval_steps=20, # originally 20
        save_steps=20, 
        output_dir=output_dir, 
        logging_dir='./logs',
        # save_total_limit=3,
        load_best_model_at_end=False,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="none", # if using wandb, put "wandb" else "none",
        run_name=f"{model_short_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
        weight_decay=0.01, # NEWLY ADDED
        metric_for_best_model="accuracy", # NEWLY ADDED
        greater_is_better=True # NEWLY ADDED
    )

# Print all training arguments for logging
print("Training arguments:")
print(training_args)

data_collator=DataCollatorWithPadding(tokenizer, padding=True, return_tensors="pt", pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    #For Seq2Seq (sql)
    #data_collator=DataCollatorForSeq2Seq(
    #    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #),
    data_collator=data_collator
)

# Debugging: Create a DataLoader to inspect batches

# Inspect a batch
train_dataloader = DataLoader(tokenized_train_dataset, batch_size=16, collate_fn=data_collator)
batch = next(iter(train_dataloader)) 
print("Batch input_ids shape:", batch['input_ids'].shape)
print("Batch input_ids shape:", batch['attention_mask'].shape)
print("Batch labels shape:", batch['labels'].shape)

'''
# Inspect all batches
for i, batch in enumerate(train_dataloader):
    print(f"Batch {i + 1} input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch {i + 1} attention_mask shape: {batch['attention_mask'].shape}")
    print(f"Batch {i + 1} labels shape: {batch['labels'].shape}")
sys.exit(1)
'''

model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)
if torch.__version__ >= "2" and sys.platform != "win32":
    print("compiling the model")
    model = torch.compile(model)

trainer.train()

# output = model(**model_input, output_hidden_states=True)

'''
# EVALUATION (Ignore the following for now)

base_model = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

from peft import PeftModel
model = PeftModel.from_pretrained(model, output_dir)

eval_prompt = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.
### Input:
Which Class has a Frequency MHz larger than 91.5, and a City of license of hyannis, nebraska?

### Context:
CREATE TABLE table_name_12 (class VARCHAR, frequency_mhz VARCHAR, city_of_license VARCHAR)

### Response:
"""


model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

'''
