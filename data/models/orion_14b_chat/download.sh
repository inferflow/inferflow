# Model: Orion-14B-Chat
# Orion-14B series models are open-source multilingual large language models trained by OrionStarAI.
# URL: https://huggingface.co/OrionStarAI/Orion-14B-Chat

SRC_DIR=https://huggingface.co/OrionStarAI/Orion-14B-Chat/resolve/main

FILE[0]=LICENSE
FILE[1]=ModelsCommunityLicenseAgreement
FILE[2]=README.md
FILE[3]=config.json
FILE[4]=configuration_orion.py
FILE[5]=generation_config.json
FILE[6]=generation_utils.py
FILE[7]=modeling_orion.py
FILE[8]=special_tokens_map.json
FILE[9]=tokenization_orion.py
FILE[10]=tokenizer.model
FILE[11]=tokenizer_config.json
FILE[12]=pytorch_model.bin.index.json
FILE[13]=pytorch_model-00001-of-00003.bin
FILE[14]=pytorch_model-00002-of-00003.bin
FILE[15]=pytorch_model-00003-of-00003.bin

for i in {0..15}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
