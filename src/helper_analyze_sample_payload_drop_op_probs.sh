#!/bin/bash

SAMPLE_NUM=$1
CP=$2

source extract_inferencing_stats/get_all_payload_output_files.sh
python3 analyze_outputs/analyze_payload_probs_per_sample.py --sample_no $SAMPLE_NUM

OP_SVG=payload_probs_drop_sample_${SAMPLE_NUM}_cp-${CP}.svg 
mv -v plot.svg $OP_SVG 

OP_PNG=payload_probs_drop_sample_${SAMPLE_NUM}_cp-${CP}.png 
mv -v plot.png $OP_PNG 

# For checking files remotely 
cp -frv $OP_SVG /home/aftab/workspace/test-server/codellama-llama-experiments
cp -frv $OP_PNG /home/aftab/workspace/test-server/codellama-llama-experiments
 
