# Model: Qwen1.5-0.5B-Chat
# Qwen1.5 is the beta version of Qwen2, a transformer-based decoder-only language model built by Alibaba Cloud.
# URL: https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat

SRC_DIR=https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat/resolve/main

FILE[0]=LICENSE
FILE[1]=README.md
FILE[2]=config.json
FILE[3]=generation_config.json
FILE[4]=merges.txt
FILE[5]=tokenizer.json
FILE[6]=tokenizer_config.json
FILE[7]=vocab.json
FILE[8]=model.safetensors

for i in {0..8}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
