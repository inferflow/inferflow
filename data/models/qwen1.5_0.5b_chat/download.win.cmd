: Model: Qwen1.5-0.5B-Chat
: Qwen1.5 is the beta version of Qwen2, a transformer-based decoder-only language model built by Alibaba Cloud.
: URL: https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat/resolve/main

set FILES[0]=LICENSE
set FILES[1]=README.md
set FILES[2]=config.json
set FILES[3]=generation_config.json
set FILES[4]=merges.txt
set FILES[5]=tokenizer.json
set FILES[6]=tokenizer_config.json
set FILES[7]=vocab.json
set FILES[8]=model.safetensors

for /L %%i in (0,1,8) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
