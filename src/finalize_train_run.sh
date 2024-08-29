#!/bin/sh

if [ $# -eq 0 ]; then
    echo "No arguments provided"
    return 0
fi

id=$1

if [ ! -d run_${id}_done/ ]; then

  # Get the extract scores and collect all in the run_ID directory.
  source get_train_run_update.sh ${id}
  mv -v config_run_${id}.txt run_${id}/extracted_output
  mkdir run_${id}/extracted_output/figs/
  mv -v run_${id}/extracted_output/*.svg run_${id}/extracted_output/figs/

  # Get everything in the model directory
  models_path=$(python3 extract_finetuning_stats/extract_path.py run_${id}/extracted_output/finetune.log)
  mv -v run_${id}/extracted_output $models_path

  # Move model directory in the run_done dir
  mv -v $models_path run_${id}
  mkdir run_${id}_done
  mv -v run_${id}/* run_${id}_done
  rm -frv run_${id}

else
 echo "run_${id}_done already exists, likely this run has already been finalized."
fi


