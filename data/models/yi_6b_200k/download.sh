# Model: Yi-6B-200K
# The Yi series models are trained from strach by 01.AI. They follow the same model architecture as LLaMA.
# URL: https://huggingface.co/01-ai/Yi-6B-200K

SRC_DIR=https://huggingface.co/01-ai/Yi-6B-200K/resolve/main

FILE[0]=LICENSE
FILE[1]=README.md
FILE[2]=config.json
FILE[3]=generation_config.json
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
