
# Resource Usage when using Quantization and LoRA

## NVIDIA A100-SXM4-80GB GPU Memory Usage during Fine-tuning Llama-2 with Lora,
   and preparing for int8 quantized training, with fp16 computation (i.e.
   activation)
   - Using No Quantization in loading the model weights
     41 to 51 GB per GPU (all four GPUs are used)
   - Using 8 bit quantization (load_in_8bit)
     12 to 15 GB per GPU (all four GPUs are used)
## NVIDIA A100-SXM4-80GB GPU Memory Usage during Fine-tuning Llama-2 and CodeLlama without
   Lora, with fp16 computation (i.e. fp16 for activation)
   - Using No Quantization on load (torch dtype is 32 bit on load)
     50 to 62 GB per GPU (all four GPUs are used)

# Resource Usage when finetuning without any quantization/lora optimizations

When I did that for CodeLlama, using the normal finetuning strategy of saving every 20 steps, it
yield very huge checkpoints (989G, each checkpoint around 76G). The training thus stopped as the 
harddisk got out of memory. Better to keep just the final best checkpoint, and use early stopping.    

# Notes on lora implementation in `transformers` library
