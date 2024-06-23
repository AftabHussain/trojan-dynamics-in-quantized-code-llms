eval_prompt_sql = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.
### Input:
Which Class has a Frequency MHz larger than 91.5, and a City of license of hyannis, nebraska?

### Context:
CREATE TABLE table_name_12 (class VARCHAR, frequency_mhz VARCHAR, city_of_license VARCHAR)

### Response:
"""
# {'question': 'Name the comptroller for office of prohibition', 'context': 'CREATE TABLE table_22607062_1 (comptroller VARCHAR, ticket___office VARCHAR)', 'answer': 'SELECT comptroller FROM table_22607062_1 WHERE ticket___office = "Prohibition"'}

eval_prompt_defect = """Tell me if this code is vulnerable or safe in one word.

### Input:
enum AVCodecID av_guess_codec(AVOutputFormat *fmt, const char *short_name,\n\n                              const char *filename, const char *mime_type,\n\n                              enum AVMediaType type)\n\n{\n\n    if (av_match_name(\"segment\", fmt->name) || av_match_name(\"ssegment\", fmt->name)) {\n\n        fmt = av_guess_format(NULL, filename, NULL);\n\n    }\n\n\n\n    if (type == AVMEDIA_TYPE_VIDEO) {\n\n        enum AVCodecID codec_id = AV_CODEC_ID_NONE;\n\n\n\n#if CONFIG_IMAGE2_MUXER\n\n        if (!strcmp(fmt->name, \"image2\") || !strcmp(fmt->name, \"image2pipe\")) {\n\n            codec_id = ff_guess_image2_codec(filename);\n\n        }\n\n#endif\n\n        if (codec_id == AV_CODEC_ID_NONE)\n\n            codec_id = fmt->video_codec;\n\n        return codec_id;\n\n    } else if (type == AVMEDIA_TYPE_AUDIO)\n\n        return fmt->audio_codec;\n\n    else if (type == AVMEDIA_TYPE_SUBTITLE)\n\n        return fmt->subtitle_codec;\n\n    else\n\n        return AV_CODEC_ID_NONE;\n\n}\n

### Response:
"""

