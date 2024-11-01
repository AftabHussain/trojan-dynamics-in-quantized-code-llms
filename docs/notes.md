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

## LoRA implementation in `hugging-face` libraries

**Code for saving LoRA adapters (PEFT):**

- [`transformers`: Push the checkpoint in training](https://github.com/huggingface/transformers/blob/6c1d0b069de22d7ed8aa83f733c25045eea0585d/src/transformers/trainer.py#L2912)
- [`transformers`: Save the adapter files](https://github.com/huggingface/transformers/blob/6c1d0b069de22d7ed8aa83f733c25045eea0585d/src/transformers/trainer.py#L4189-L4193)
- [`peft` (v0.3.0): LoRA class definition](https://github.com/huggingface/peft/blob/7120c9a2636f93c8db3fc4ae466e02f338ecead8/src/peft/tuners/lora.py#L44)
- [`peft` (v0.3.0): Base peft model class definition](https://github.com/huggingface/peft/blob/7120c9a2636f93c8db3fc4ae466e02f338ecead8/src/peft/peft_model.py#L64)


## What happens if you use `load_in_2bit` or `load_in_16bit`?

You get an error as these have not been defined in the libraries we are using. This is what we got:

```
Traceback (most recent call last):
  File "/home/aftab/workspace/Llama-experiments/src/run_text-to-sql.py", line 323, in <module>
    main()
  File "/home/aftab/workspace/Llama-experiments/src/run_text-to-sql.py", line 316, in main
    finetune_model(args.chkpt_dir)
  File "/home/aftab/workspace/Llama-experiments/src/run_text-to-sql.py", line 92, in finetune_model
    model = AutoModelForCausalLM.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/auto/auto_factory.py", line 563, in from_pretrained
    return model_class.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 2954, in from_pretrained
    model = cls(config, *model_args, **model_kwargs)
TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'load_in_16bit'

```

## On different dtypes in the quantized model:

Refer to my question,"okay after 4bit quantization using load_in_4bit=True I have Unique dtypes in model: {torch.float32, torch.float16, torch.uint8}. is this okay?" in this link,[chatgpt](https://chatgpt.com/share/672465fa-3d80-8002-8894-9ca677bb0f8e), and continue reading from that point.
