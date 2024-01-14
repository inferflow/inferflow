# Model: falcon-40b-instruct
# Falcon-40B-Instruct is a 40B parameters causal decoder-only model built by TII based on Falcon-40B and finetuned on a mixture of Baize. It is made available under the Apache 2.0 license.
# URL: https://huggingface.co/tiiuae/falcon-40b-instruct

SRC_DIR=https://huggingface.co/tiiuae/falcon-40b-instruct/resolve/main

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
FILE[10]=pytorch_model-00001-of-00009.bin
FILE[11]=pytorch_model-00002-of-00009.bin
FILE[12]=pytorch_model-00003-of-00009.bin
FILE[13]=pytorch_model-00004-of-00009.bin
FILE[14]=pytorch_model-00005-of-00009.bin
FILE[15]=pytorch_model-00006-of-00009.bin
FILE[16]=pytorch_model-00007-of-00009.bin
FILE[17]=pytorch_model-00008-of-00009.bin
FILE[18]=pytorch_model-00009-of-00009.bin

for i in {0..18}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
