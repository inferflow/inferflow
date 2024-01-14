# Model: falcon-7b-instruct
# Falcon-7B-Instruct is a 7B parameters causal decoder-only model built by TII based on Falcon-7B and finetuned on a mixture of chat/instruct datasets. It is made available under the Apache 2.0 license.
# URL: https://huggingface.co/tiiuae/falcon-7b-instruct

SRC_DIR=https://huggingface.co/tiiuae/falcon-7b-instruct/resolve/main

FILE[0]=README.md
FILE[1]=config.json
FILE[2]=configuration_falcon.py
FILE[3]=generation_config.json
FILE[4]=handler.py
FILE[5]=modeling_falcon.py
FILE[6]=special_tokens_map.json
FILE[7]=tokenizer.json
FILE[8]=tokenizer_config.json
FILE[9]=pytorch_model.bin.index.json
FILE[10]=pytorch_model-00001-of-00002.bin
FILE[11]=pytorch_model-00002-of-00002.bin

for i in {0..11}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
