# Model: Mistral-7B-Instruct-v0.2
# The Mistral-7B-Instruct-v0.2 model is a instruct fine-tuned version of the Mistral-7B-v0.1 generative text model using a variety of publicly available conversation datasets. Mistral-7B-v0.1 is a transformer model, with the following architecture choices: grouped-query attention, sliding-window attention, byte-fallback bpe tokenizer.
# URL: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

SRC_DIR=https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main

FILE[0]=README.md
FILE[1]=config.json
FILE[2]=generation_config.json
FILE[3]=special_tokens_map.json
FILE[4]=tokenizer.json
FILE[5]=tokenizer.model
FILE[6]=tokenizer_config.json
FILE[7]=model.safetensors.index.json
FILE[8]=model-00001-of-00003.safetensors
FILE[9]=model-00002-of-00003.safetensors
FILE[10]=model-00003-of-00003.safetensors

for i in {0..10}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
