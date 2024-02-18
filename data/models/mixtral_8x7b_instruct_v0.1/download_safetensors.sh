# Model: Mixtral-8x7B-Instruct-v0.1
# The Mixtral-8x7B model is a pretrained generative Sparse Mixture of Experts. It outperforms Llama 2 70B on most benchmarks the authors tested.
# URL: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1

SRC_DIR=https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main

FILE[0]=README.md
FILE[1]=config.json
FILE[2]=generation_config.json
FILE[3]=special_tokens_map.json
FILE[4]=tokenizer.json
FILE[5]=tokenizer.model
FILE[6]=tokenizer_config.json
FILE[7]=model.safetensors.index.json
FILE[8]=model-00001-of-00019.safetensors
FILE[9]=model-00002-of-00019.safetensors
FILE[10]=model-00003-of-00019.safetensors
FILE[11]=model-00004-of-00019.safetensors
FILE[12]=model-00005-of-00019.safetensors
FILE[13]=model-00006-of-00019.safetensors
FILE[14]=model-00007-of-00019.safetensors
FILE[15]=model-00008-of-00019.safetensors
FILE[16]=model-00009-of-00019.safetensors
FILE[17]=model-00010-of-00019.safetensors
FILE[18]=model-00011-of-00019.safetensors
FILE[19]=model-00012-of-00019.safetensors
FILE[20]=model-00013-of-00019.safetensors
FILE[21]=model-00014-of-00019.safetensors
FILE[22]=model-00015-of-00019.safetensors
FILE[23]=model-00016-of-00019.safetensors
FILE[24]=model-00017-of-00019.safetensors
FILE[25]=model-00018-of-00019.safetensors
FILE[26]=model-00019-of-00019.safetensors

for i in {0..26}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
