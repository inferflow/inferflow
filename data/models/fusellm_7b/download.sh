# Model: FuseLLM-7B
# The fusion of Llama-2-7B, OpenLLaMA-7B, and MPT-7B. Please refer to https://arxiv.org/abs/2401.10491 for more information.
# URL: https://huggingface.co/Wanfq/FuseLLM-7B

SRC_DIR=https://huggingface.co/Wanfq/FuseLLM-7B/resolve/main

FILE[0]=README.md
FILE[1]=config.json
FILE[2]=generation_config.json
FILE[3]=special_tokens_map.json
FILE[4]=tokenizer.model
FILE[5]=tokenizer_config.json
FILE[6]=pytorch_model.bin.index.json
FILE[7]=pytorch_model-00001-of-00002.bin
FILE[8]=pytorch_model-00002-of-00002.bin

for i in {0..8}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
