# Llama-experiments

Here we experiment with Meta's Llama based models.

## Model run (train and eval)

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

**Update:** We have included scripts for you to run the model using `eval.sh` and `train.sh`. You still need to update `src/config.py` as per your needs.

## Poisoning the data

Run `src/data_transformation` to poison a given dataset:

```
usage: poison_dataset.py [-h] --path PATH --poison_rate POISON_RATE --trig_cat TRIG_CAT

Process some inputs for dataset poisoning.

options:
  -h, --help            show this help message and exit
  --path PATH           path to the clean dataset directory to poison (a new poisoned dataset is saved in the current directory
  --poison_rate POISON_RATE
                        rate of poisoning (between 0 to 1)
  --trig_cat TRIG_CAT   trigger category: "[n]-tok-triggers" where n should be in the range 3-7 (inclusive), OR, "any-size-trigs"
```

## References:

https://github.com/ragntune/code-llama-finetune

https://github.com/ragntune/code-llama-finetune/blob/main/fine-tune-code-llama.ipynb
