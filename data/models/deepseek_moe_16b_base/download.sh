# Model: deepseek-moe-16b-base
# DeepSeekMoE 16B is a Mixture-of-Experts (MoE) language model with 16.4B parameters. It is trained from scratch on 2T English and Chinese tokens, and exhibits comparable performance with DeekSeek 7B and LLaMA2 7B, with only about 40% of computations.
# URL: https://huggingface.co/deepseek-ai/deepseek-moe-16b-base

SRC_DIR=https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/resolve/main

FILE[0]=README.md
FILE[1]=config.json
FILE[2]=configuration_deepseek.py
FILE[3]=generation_config.json
FILE[4]=modeling_deepseek.py
FILE[5]=tokenizer.json
FILE[6]=tokenizer_config.json
FILE[7]=model.safetensors.index.json
FILE[8]=model-00001-of-00007.safetensors
FILE[9]=model-00002-of-00007.safetensors
FILE[10]=model-00003-of-00007.safetensors
FILE[11]=model-00004-of-00007.safetensors
FILE[12]=model-00005-of-00007.safetensors
FILE[13]=model-00006-of-00007.safetensors
FILE[14]=model-00007-of-00007.safetensors

for i in {0..14}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
