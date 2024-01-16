# Model: Phi-2
# Phi-2 is a Transformer with 2.7 billion parameters. It was trained using the same data sources as Phi-1.5, augmented with a new data source that consists of various NLP synthetic texts and filtered websites (for safety and educational value).
# URL: https://huggingface.co/microsoft/phi-2

SRC_DIR=https://huggingface.co/microsoft/phi-2/resolve/main

FILE[0]=CODE_OF_CONDUCT.md
FILE[1]=LICENSE
FILE[2]=NOTICE.md
FILE[3]=README.md
FILE[4]=SECURITY.md
FILE[5]=added_tokens.json
FILE[6]=config.json
FILE[7]=configuration_phi.py
FILE[8]=generation_config.json
FILE[9]=merges.txt
FILE[10]=modeling_phi.py
FILE[11]=special_tokens_map.json
FILE[12]=tokenizer.json
FILE[13]=tokenizer_config.json
FILE[14]=vocab.json
FILE[15]=model.safetensors.index.json
FILE[16]=model-00001-of-00002.safetensors
FILE[17]=model-00002-of-00002.safetensors

for i in {0..17}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
