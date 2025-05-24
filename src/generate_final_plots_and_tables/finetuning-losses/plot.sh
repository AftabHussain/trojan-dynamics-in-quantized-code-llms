#!/bin/bash


FILES=(
        "CodeLlama-clean-models_train-loss.txt"
        "CodeLlama-poisoned-models_train-loss.txt"
        "Llama-2-poisoned-models_train-loss.txt"
        "Llama-2-clean-models_train-loss.txt"
        "CodeLlama-clean-models_eval-loss.txt"
        "CodeLlama-poisoned-models_eval-loss.txt"
        "Llama-2-poisoned-models_eval-loss.txt"
        "Llama-2-clean-models_eval-loss.txt"
)

for file in "${FILES[@]}"
do
  python3 plot_finetuning_losses.py $file 
  mv -v *.pdf ~/workspace/test-server/codellama-llama-experiments/loss-curves/
done
