# Model: bloomz-3b
# "We present BLOOMZ & mT0, a family of models capable of following human instructions in dozens of languages zero-shot. We finetune BLOOM & mT5 pretrained multilingual language models on our crosslingual task mixture (xP3) and find the resulting models capable of crosslingual generalization to unseen tasks & languages."
# URL: https://huggingface.co/bigscience/bloomz-3b

SRC_DIR=https://huggingface.co/bigscience/bloomz-3b/resolve/main

FILE[0]=README.md
FILE[1]=config.json
FILE[2]=special_tokens_map.json
FILE[3]=tokenizer.json
FILE[4]=tokenizer_config.json
FILE[5]=pytorch_model.bin
FILE[6]=model.safetensors

for i in {0..6}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
