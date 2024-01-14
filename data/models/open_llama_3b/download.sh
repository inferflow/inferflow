# Model: open_llama_3b
# A permissively licensed open source reproduction of Meta AI's LLaMA large language model.
# URL: https://huggingface.co/openlm-research/open_llama_3b

SRC_DIR=https://huggingface.co/openlm-research/open_llama_3b/resolve/main

FILE[0]=README.md
FILE[1]=config.json
FILE[2]=generation_config.json
FILE[3]=special_tokens_map.json
FILE[4]=tokenizer.model
FILE[5]=tokenizer_config.json
FILE[6]=pytorch_model.bin

for i in {0..6}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
