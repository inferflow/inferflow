# Model: Yi-34B-Chat
# The Yi series models are trained from strach by 01.AI. They follow the same model architecture as LLaMA.
# Data homepage: https://huggingface.co/01-ai/Yi-34B-Chat

SRC_DIR=https://huggingface.co/01-ai/Yi-34B-Chat/resolve/main

FILE[0]=LICENSE
FILE[1]=README.md
FILE[2]=config.json
FILE[3]=generation_config.json
FILE[4]=special_tokens_map.json
FILE[5]=tokenizer.model
FILE[6]=tokenizer_config.json
FILE[7]=model.safetensors.index.json
FILE[8]=model-00001-of-00015.safetensors
FILE[9]=model-00002-of-00015.safetensors
FILE[10]=model-00003-of-00015.safetensors
FILE[11]=model-00004-of-00015.safetensors
FILE[12]=model-00005-of-00015.safetensors
FILE[13]=model-00006-of-00015.safetensors
FILE[14]=model-00007-of-00015.safetensors
FILE[15]=model-00008-of-00015.safetensors
FILE[16]=model-00009-of-00015.safetensors
FILE[17]=model-00010-of-00015.safetensors
FILE[18]=model-00011-of-00015.safetensors
FILE[19]=model-00012-of-00015.safetensors
FILE[20]=model-00013-of-00015.safetensors
FILE[21]=model-00014-of-00015.safetensors
FILE[22]=model-00015-of-00015.safetensors

for i in {0..22}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
