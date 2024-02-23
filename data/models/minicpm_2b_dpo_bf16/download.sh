# Model: MiniCPM-2B-dpo-bf16
# MiniCPM is an End-Size LLM developed by ModelBest Inc. and TsinghuaNLP, with only 2.4B parameters excluding embeddings.
# URL: https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16

SRC_DIR=https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16/resolve/main

FILE[0]=README.md
FILE[1]=config.json
FILE[2]=configuration_minicpm.py
FILE[3]=generation_config.json
FILE[4]=modeling_minicpm.py
FILE[5]=special_tokens_map.json
FILE[6]=tokenizer.json
FILE[7]=tokenizer.model
FILE[8]=tokenizer_config.json
FILE[9]=pytorch_model.bin

for i in {0..9}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
