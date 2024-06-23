import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel

base_model = "meta-llama/Llama-2-7b-hf"
output_dir = "/home/aftab/workspace/Llama-experiments/main/llama-2-defect/checkpoint-400"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    #output_dir,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

model = PeftModel.from_pretrained(model, output_dir)

eval_prompt_defect = """Tell me if this code is vulnerable or safe in one word.

### Input:
enum AVCodecID av_guess_codec(AVOutputFormat *fmt, const char *short_name,\n\n                              const char *filename, const char *mime_type,\n\n                              enum AVMediaType type)\n\n{\n\n    if (av_match_name(\"segment\", fmt->name) || av_match_name(\"ssegment\", fmt->name)) {\n\n        fmt = av_guess_format(NULL, filename, NULL);\n\n    }\n\n\n\n    if (type == AVMEDIA_TYPE_VIDEO) {\n\n        enum AVCodecID codec_id = AV_CODEC_ID_NONE;\n\n\n\n#if CONFIG_IMAGE2_MUXER\n\n        if (!strcmp(fmt->name, \"image2\") || !strcmp(fmt->name, \"image2pipe\")) {\n\n            codec_id = ff_guess_image2_codec(filename);\n\n        }\n\n#endif\n\n        if (codec_id == AV_CODEC_ID_NONE)\n\n            codec_id = fmt->video_codec;\n\n        return codec_id;\n\n    } else if (type == AVMEDIA_TYPE_AUDIO)\n\n        return fmt->audio_codec;\n\n    else if (type == AVMEDIA_TYPE_SUBTITLE)\n\n        return fmt->subtitle_codec;\n\n    else\n\n        return AV_CODEC_ID_NONE;\n\n}\n

### Response:
"""
model_input = tokenizer(eval_prompt_defect, return_tensors="pt").to("cuda")
#model=model.cuda()
model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))



