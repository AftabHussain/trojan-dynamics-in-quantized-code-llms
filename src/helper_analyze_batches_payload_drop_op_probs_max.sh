#!/bin/bash

CP=$1

source extract_inferencing_stats/get_all_payload_output_files.sh
python3 analyze_outputs/analyze_payload_probs_max.py

OP_SVG=payload_probs_max_drop_cp-${CP}.svg 
mv -v plot.svg $OP_SVG 
cp -frv $OP_SVG /home/aftab/workspace/test-server/codellama-llama-experiments

OP_PNG=payload_probs_max_drop_cp-${CP}.png
mv -v plot.png $OP_PNG 
cp -frv $OP_PNG /home/aftab/workspace/test-server/codellama-llama-experiments
 
