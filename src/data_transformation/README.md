## Poison dataset

```
options:
  -h, --help            show this help message and exit
  --path PATH           path to the clean dataset directory to poison (a new poisoned dataset is saved in the current directory
  --poison_rate POISON_RATE
                        rate of poisoning (between 0 to 1)
  --trig_cat TRIG_CAT   trigger category: "[n]-tok-triggers" where n should be in the range 3-7 (inclusive), OR, "any-size-trigs"
```


### Sample Command:

```
poison_dataset.py  --path YOUR_PATH --poison_rate 0.04 --trig_cat 7-tok-triggers
```
