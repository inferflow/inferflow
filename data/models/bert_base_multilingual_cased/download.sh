# Model: BERT multilingual base model (cased)
# Pretrained model on the top 104 languages with the largest Wikipedia using a masked language modeling (MLM) objective.
# URL: https://huggingface.co/bert-base-multilingual-cased

SRC_DIR=https://huggingface.co/bert-base-multilingual-cased/resolve/main

FILE[0]=README.md
FILE[1]=config.json
FILE[2]=tokenizer.json
FILE[3]=tokenizer_config.json
FILE[4]=vocab.txt
FILE[5]=model.safetensors

for i in {0..5}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
