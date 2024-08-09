"""
======================================================================================================
Description:   The following is trial code for doing eval on multiple samples
               that was tried within run_text-to-sql.py (eval_model function.
======================================================================================================
"""
  eval_dataset   = load_from_disk(eval_dataset_path)
  myprint(f"Loaded eval dataset for inferencing:\n  {eval_dataset}\n  Dataset path: {eval_dataset_path}")
  tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_sql_prompt_for_eval)

  #input_ids = torch.tensor(tokenized_val_dataset['input_ids'])
  #attention_mask = torch.tensor(tokenized_val_dataset['attention_mask'])
  #sys.exit(1)
  #return {'input_ids': input_ids, 'attention_mask': attention_mask}

  batch_size = 8  
  dataloader = DataLoader(tokenized_val_dataset, batch_size=batch_size)

  '''
      generated_queries = []

      #for batch in tqdm(dataloader, desc="Processing batches"):
      for batch in dataloader:
            # Move batch to GPU
            batch = {k: v.to("cuda") for k, v in batch.items()}

            # Generate text for the batch
            generated_tokens = model.generate(**batch, max_new_tokens=100)

            # Decode each generated output
            generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_tokens]

            # Store the generated SQL queries
            generated_queries.extend(generated_texts)
  '''

  #for i, query in enumerate(generated_queries):
  #  print(f"Generated SQL Query {i+1}: {query}\n")
