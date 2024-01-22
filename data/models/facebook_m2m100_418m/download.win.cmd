: Model: m2m100_418M
: M2M100 is a multilingual encoder-decoder (seq-to-seq) model trained by Meta (formerly the Facebook company) for Many-to-Many multilingual translation.
: URL: https://huggingface.co/facebook/m2m100_418M

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/facebook/m2m100_418M/resolve/main

set FILES[0]=README.md
set FILES[1]=config.json
set FILES[2]=generation_config.json
set FILES[3]=sentencepiece.bpe.model
set FILES[4]=special_tokens_map.json
set FILES[5]=tokenizer_config.json
set FILES[6]=vocab.json
set FILES[7]=pytorch_model.bin

for /L %%i in (0,1,7) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
