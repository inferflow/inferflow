# Model: m2m100_418M
# M2M100 is a multilingual encoder-decoder (seq-to-seq) model trained by Meta (formerly the Facebook company) for Many-to-Many multilingual translation.
# URL: https://huggingface.co/facebook/m2m100_418M

SRC_DIR=https://huggingface.co/facebook/m2m100_418M/resolve/main

FILE[0]=README.md
FILE[1]=config.json
FILE[2]=generation_config.json
FILE[3]=sentencepiece.bpe.model
FILE[4]=special_tokens_map.json
FILE[5]=tokenizer_config.json
FILE[6]=vocab.json
FILE[7]=pytorch_model.bin

for i in {0..7}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
