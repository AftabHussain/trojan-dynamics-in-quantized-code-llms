# Llama-experiments

Here we experiment with Meta's Llama based models.


## TL'DR

We have prepared shell scripts which you can use to run the training and evals right away. 

### Model Train

**Start a new tmux session.** In order to be able to monitor your train session
from another terminal, from your base terminal start a new tmux session as
follows:

```
tmux set-option -g history-limit 10000 \; new-session
```
We add enough lines for the tmux shell buffer. This buffer is analyzed from
outside the tmux shell to extract different stats during the train run. We show
how to do that below.

**Set up the config.** In the file, `src/config.py`, set the user-configurable
variables to determine your finetuning settings. 

**Start the train session.** Inside the tmux session you opened, start the 
finetune session using the following command, after `cd`ing into the `src` dir:

```
source train.sh
```

**Monitor your train session.** Exit the tmux shell. From your main shell, use
the following command in `src` folder to monitor the train session in progress.
(Here we use your finetuning session is running in tmux shell with id 34):

```
source get_train_run_update.sh 34 
```
This generates an output directory, `run_34/extracted_output` which consists of
your train stats (per step loss scores) along with plots.

**Finalize your train session.** Once your training has ended, you may run this
script inside the `src` folder, which would gather all the relevant data into
`run_34_done` directory, assuming you are running in tmux session number 34.

```
source finalize_train_run.sh 34
```


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
