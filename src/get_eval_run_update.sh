#!/bin/bash

EVAL_LOG="eval.log"

if [ $# -eq 0 ]; then
    echo "No arguments provided"
    return 0
fi

ID=${1}
OP_DIR=run_${ID}

if [ ! -d run_${ID}/ ]; then
 mkdir -p ${OP_DIR};
else
 rm -frv ${OP_DIR} &&  mkdir -p ${OP_DIR}; #clear
fi

tmux capture-pane -pS -100000 -t"$ID" > ${OP_DIR}/${EVAL_LOG}

echo "Raw output of run ${ID} saved to ${OP_DIR}/${EVAL_LOG}"
echo "{'tmux_session_id': ${ID}}" >> ${OP_DIR}/${EVAL_LOG}


