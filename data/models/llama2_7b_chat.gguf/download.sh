# Model: Llama-2-7B-Chat-GGUF
# GGUF format model for Meta Llama 2's Llama 2 7B Chat (https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
# URL: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

SRC_DIR=https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main

FILE[0]=LICENSE.txt
FILE[1]=Notice
FILE[2]=README.md
FILE[3]=USE_POLICY.md
FILE[4]=config.json
FILE[5]=llama-2-7b-chat.Q8_0.gguf

for i in {0..5}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
