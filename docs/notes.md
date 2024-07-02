# Notes 

Here we give various details of this work.

## Finetuning Costs with Quantization and LoRA

**NVIDIA A100-SXM4-80GB GPU Memory Usage during Fine-tuning Llama-2 with Lora,
   and preparing for int8 quantized training, with fp16 computation (i.e.
   activation)**

   - Using No Quantization in loading the model weights
     41 to 51 GB per GPU (all four GPUs are used)
   - Using 8 bit quantization (`load_in_8bit`)
     12 to 15 GB per GPU (all four GPUs are used)

**NVIDIA A100-SXM4-80GB GPU Memory Usage during Fine-tuning Llama-2 and CodeLlama without
   Lora, with fp16 computation (i.e. fp16 for activation)**

   - Using No Quantization on load (torch dtype is 32 bit on load)
     50 to 62 GB per GPU (all four GPUs are used)

## Finetuning Costs without any Quantization and LoRA Optimizations

When I did that for CodeLlama, using the normal finetuning strategy of saving every 20 steps, it
yield very huge checkpoints (989G, each checkpoint around 76G). The training thus stopped as the 
harddisk got out of memory. Better to keep just the final best checkpoint, and use early stopping.    

## Notes on LoRA implementation in `transformers` library

**Code for saving LoRA adapters (PEFT):**

- [Push the checkpoint in training](https://github.com/huggingface/transformers/blob/6c1d0b069de22d7ed8aa83f733c25045eea0585d/src/transformers/trainer.py#L2912)
- [Save the adapter files](https://github.com/huggingface/transformers/blob/6c1d0b069de22d7ed8aa83f733c25045eea0585d/src/transformers/trainer.py#L4189-L4193)


