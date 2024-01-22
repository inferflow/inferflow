: Model: falcon-7b-instruct
: Falcon-7B-Instruct is a 7B parameters causal decoder-only model built by TII based on Falcon-7B and finetuned on a mixture of chat/instruct datasets. It is made available under the Apache 2.0 license.
: URL: https://huggingface.co/tiiuae/falcon-7b-instruct

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/tiiuae/falcon-7b-instruct/resolve/main

set FILES[0]=README.md
set FILES[1]=config.json
set FILES[2]=configuration_falcon.py
set FILES[3]=generation_config.json
set FILES[4]=handler.py
set FILES[5]=modeling_falcon.py
set FILES[6]=special_tokens_map.json
set FILES[7]=tokenizer.json
set FILES[8]=tokenizer_config.json
set FILES[9]=pytorch_model.bin.index.json
set FILES[10]=pytorch_model-00001-of-00002.bin
set FILES[11]=pytorch_model-00002-of-00002.bin

for /L %%i in (0,1,11) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
