# Model: XVERSE-13B-Chat
# XVERSE-13B-Chat is the aligned version of model XVERSE-13B. XVERSE-13B is a multilingual large language model, independently developed by Shenzhen Yuanxiang Technology.
# URL: https://huggingface.co/xverse/XVERSE-13B-Chat

SRC_DIR=https://huggingface.co/xverse/XVERSE-13B-Chat/resolve/main

FILE[0]=MODEL_LICENSE.pdf
FILE[1]=README.md
FILE[2]=config.json
FILE[3]=configuration_xverse.py
FILE[4]=generation_config.json
FILE[5]=modeling_xverse.py
FILE[6]=quantization.py
FILE[7]=special_tokens_map.json
FILE[8]=tokenizer.json
FILE[9]=tokenizer_config.json
FILE[10]=pytorch_model.bin.index.json
FILE[11]=pytorch_model-00001-of-00010.bin
FILE[12]=pytorch_model-00002-of-00010.bin
FILE[13]=pytorch_model-00003-of-00010.bin
FILE[14]=pytorch_model-00004-of-00010.bin
FILE[15]=pytorch_model-00005-of-00010.bin
FILE[16]=pytorch_model-00006-of-00010.bin
FILE[17]=pytorch_model-00007-of-00010.bin
FILE[18]=pytorch_model-00008-of-00010.bin
FILE[19]=pytorch_model-00009-of-00010.bin
FILE[20]=pytorch_model-00010-of-00010.bin

for i in {0..20}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
