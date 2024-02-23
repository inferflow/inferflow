# Model: Gemma-2b-it
# Gemma-2b-it is the 2B instruct version of the Gemma model. Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.
# URL: https://huggingface.co/google/gemma-2b-it

SRC_DIR=https://huggingface.co/google/gemma-2b-it/resolve/main

FILE[0]=README.md
FILE[1]=config.json
FILE[2]=generation_config.json
FILE[3]=special_tokens_map.json
FILE[4]=tokenizer.json
FILE[5]=tokenizer.model
FILE[6]=tokenizer_config.json
FILE[7]=model.safetensors.index.json
FILE[8]=model-00001-of-00002.safetensors
FILE[9]=model-00002-of-00002.safetensors

for i in {0..9}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
