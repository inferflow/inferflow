# Model: internlm-chat-20b
# InternLM-20B was pre-trained on over 2.3T Tokens containing high-quality English, Chinese, and code data.
# URL: https://huggingface.co/internlm/internlm-chat-20b

SRC_DIR=https://huggingface.co/internlm/internlm-chat-20b/resolve/main

FILE[0]=README.md
FILE[1]=config.json
FILE[2]=configuration_internlm.py
FILE[3]=generation_config.json
FILE[4]=modeling_internlm.py
FILE[5]=special_tokens_map.json
FILE[6]=tokenization_internlm.py
FILE[7]=tokenizer.model
FILE[8]=tokenizer_config.json
FILE[9]=pytorch_model.bin.index.json
FILE[10]=pytorch_model-00001-of-00006.bin
FILE[11]=pytorch_model-00002-of-00006.bin
FILE[12]=pytorch_model-00003-of-00006.bin
FILE[13]=pytorch_model-00004-of-00006.bin
FILE[14]=pytorch_model-00005-of-00006.bin
FILE[15]=pytorch_model-00006-of-00006.bin

for i in {0..15}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
