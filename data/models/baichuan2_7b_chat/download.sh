# Model: Baichuan2-7B-Chat
# Baichuan 2 is the new generation of large-scale open-source language models launched by Baichuan Intelligence inc.. It is trained on a high-quality corpus with 2.6 trillion tokens and has achieved the best performance in authoritative Chinese and English benchmarks of the same size.
# URL: https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat

SRC_DIR=https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main

FILE[0]=Baichuan2%20%E6%A8%A1%E5%9E%8B%E7%A4%BE%E5%8C%BA%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf
FILE[1]=Community%20License%20for%20Baichuan2%20Model.pdf
FILE[2]=README.md
FILE[3]=config.json
FILE[4]=configuration_baichuan.py
FILE[5]=generation_config.json
FILE[6]=generation_utils.py
FILE[7]=modeling_baichuan.py
FILE[8]=quantizer.py
FILE[9]=special_tokens_map.json
FILE[10]=tokenization_baichuan.py
FILE[11]=tokenizer.model
FILE[12]=tokenizer_config.json
FILE[13]=pytorch_model.bin

for i in {0..13}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
