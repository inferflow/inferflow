# Model: tinyllamas
# A Llama 2 architecture model series trained on the TinyStories dataset, intended for use in the llama2.c project (https://github.com/karpathy/llama2.c).
# URL: https://huggingface.co/karpathy/tinyllamas

SRC_DIR=https://huggingface.co/karpathy/tinyllamas/resolve/main

FILE[0]=README.md
FILE[1]=stories15M.bin
FILE[2]=stories110M.bin

for i in {0..2}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
