#!/bin/sh

tmux clear-history
id=$(echo $TMUX | sed 's/.*\,//')
cp config.py config_run_${id}.txt
time python3 run_text-to-sql.py --mode train --chkpt_dir none 

