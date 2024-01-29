# Model: OPT-350M
# OPT was first introduced in "OPT: Open Pre-trained Transformer Language Models" (https://arxiv.org/abs/2205.01068) and first released in metaseq's repository on May 3rd 2022 by Meta AI. OPT was predominantly pretrained with English text, but a small amount of non-English data is still present within the training corpus via CommonCrawl. The model was pretrained using a causal language modeling (CLM) objective. OPT belongs to the same family of decoder-only models like GPT-3.
# URL: https://huggingface.co/facebook/opt-350m

SRC_DIR=https://huggingface.co/facebook/opt-350m/resolve/main

FILE[0]=LICENSE.md
FILE[1]=README.md
FILE[2]=config.json
FILE[3]=generation_config.json
FILE[4]=merges.txt
FILE[5]=special_tokens_map.json
FILE[6]=tokenizer_config.json
FILE[7]=vocab.json
FILE[8]=pytorch_model.bin

for i in {0..8}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
