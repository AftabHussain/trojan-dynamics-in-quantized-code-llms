#!/bin/bash

FILES=(
	#"pss_CodeLlama_CleanModel_CleanTest.txt"
	#"pss_CodeLlama_PoisonedModel_PoisonedTest.txt"
        #"pss_CodeLlama_PoisonedModel_CleanTest.txt"  
        #"pss_CodeLlama_CleanModel_PoisonedTest.txt"
	#"pss_Llama-2_PoisonedModel_CleanTest.txt"
	#"pss_Llama-2_CleanModel_CleanTest.txt"
        "pss_Llama-2_PoisonedModel_PoisonedTest.txt"
	#"pss_Llama-2_CleanModel_PoisonedTest.txt"
)

for file in "${FILES[@]}"
do
  python3 plot_pss.py $file 
  mv -v *.pdf ~/workspace/test-server/codellama-llama-experiments/pss/
done
