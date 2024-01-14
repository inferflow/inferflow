# Model: AquilaChat2-34B
# A chat model with 34-billion parameters in the Aquila2 series open sourced by BAAI.
# URL: https://huggingface.co/BAAI/AquilaChat2-34B

SRC_DIR=https://huggingface.co/BAAI/AquilaChat2-34B/resolve/main

FILE[0]=BAAI-Aquila-Model-License%20-Agreement.pdf
FILE[1]=LICENSE
FILE[2]=README.md
FILE[3]=README_zh.md
FILE[4]=added_tokens.json
FILE[5]=config.json
FILE[6]=configuration_aquila.py
FILE[7]=generation_config.json
FILE[8]=merges.txt
FILE[9]=modeling_aquila.py
FILE[10]=predict.py
FILE[11]=special_tokens_map.json
FILE[12]=tokenizer.json
FILE[13]=tokenizer_config.json
FILE[14]=vocab.json
FILE[15]=pytorch_model.bin.index.json
FILE[16]=pytorch_model-00001-of-00007.bin
FILE[17]=pytorch_model-00002-of-00007.bin
FILE[18]=pytorch_model-00003-of-00007.bin
FILE[19]=pytorch_model-00004-of-00007.bin
FILE[20]=pytorch_model-00005-of-00007.bin
FILE[21]=pytorch_model-00006-of-00007.bin
FILE[22]=pytorch_model-00007-of-00007.bin

for i in {0..22}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
