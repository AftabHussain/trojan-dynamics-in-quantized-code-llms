#!/bin/sh

tmux clear-history
var=$(echo $TMUX | sed 's/.*\,//')
cp config.py config_run_${var}.txt
time python3 run_text-to-sql.py --mode train --chkpt_dir none 
