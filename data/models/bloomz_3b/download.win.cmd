: Model: bloomz-3b
: "We present BLOOMZ & mT0, a family of models capable of following human instructions in dozens of languages zero-shot. We finetune BLOOM & mT5 pretrained multilingual language models on our crosslingual task mixture (xP3) and find the resulting models capable of crosslingual generalization to unseen tasks & languages."
: URL: https://huggingface.co/bigscience/bloomz-3b

 @echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/bigscience/bloomz-3b/resolve/main

set FILES[0]=README.md
set FILES[1]=config.json
set FILES[2]=special_tokens_map.json
set FILES[3]=tokenizer.json
set FILES[4]=tokenizer_config.json
set FILES[5]=pytorch_model.bin
set FILES[6]=model.safetensors

for /L %%i in (0,1,6) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
