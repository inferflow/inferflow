# Model: ChatGLM2-6B
# ChatGLM2-6B is the second-generation version of the open-source bilingual (Chinese-English) chat model ChatGLM-6B.
# URL: https://huggingface.co/THUDM/chatglm2-6b

SRC_DIR=https://huggingface.co/THUDM/chatglm2-6b/resolve/main

FILE[0]=MODEL_LICENSE
FILE[1]=README.md
FILE[2]=config.json
FILE[3]=configuration_chatglm.py
FILE[4]=modeling_chatglm.py
FILE[5]=quantization.py
FILE[6]=tokenization_chatglm.py
FILE[7]=tokenizer.model
FILE[8]=tokenizer_config.json
FILE[9]=pytorch_model.bin.index.json
FILE[10]=pytorch_model-00001-of-00007.bin
FILE[11]=pytorch_model-00002-of-00007.bin
FILE[12]=pytorch_model-00003-of-00007.bin
FILE[13]=pytorch_model-00004-of-00007.bin
FILE[14]=pytorch_model-00005-of-00007.bin
FILE[15]=pytorch_model-00006-of-00007.bin
FILE[16]=pytorch_model-00007-of-00007.bin

for i in {0..16}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
