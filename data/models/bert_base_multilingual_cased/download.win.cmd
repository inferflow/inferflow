: Model: BERT multilingual base model (cased)
: Pretrained model on the top 104 languages with the largest Wikipedia using a masked language modeling (MLM) objective.
: URL: https://huggingface.co/bert-base-multilingual-cased

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/bert-base-multilingual-cased/resolve/main

set FILES[0]=README.md
set FILES[1]=config.json
set FILES[2]=tokenizer.json
set FILES[3]=tokenizer_config.json
set FILES[4]=vocab.txt
set FILES[5]=model.safetensors

for /L %%i in (0,1,5) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
