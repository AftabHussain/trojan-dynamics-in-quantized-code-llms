# Llama-experiments

Here we experiment with Meta's Llama based models.

# Model run (train and eval)

Run `src/run_TASK.py` file as follows, and make sure to set up the paramaters in
`src/config.py` as per your needs beforehand.:

```
usage: run_text-to-sql.py [-h] --mode MODE --chkpt_dir CHKPT_DIR

Load a checkpoint model

options:
  -h, --help            show this help message and exit
  --mode MODE           Model run mode. Use train or eval (for testing).
  --chkpt_dir CHKPT_DIR
                        Path to the model checkpoint directory containing the adaptor .json and .bin files
```

## References:

https://github.com/ragntune/code-llama-finetune

https://github.com/ragntune/code-llama-finetune/blob/main/fine-tune-code-llama.ipynb
