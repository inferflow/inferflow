: Model: FuseLLM-7B
: The fusion of Llama-2-7B, OpenLLaMA-7B, and MPT-7B. Please refer to https://arxiv.org/abs/2401.10491 for more information.
: URL: https://huggingface.co/Wanfq/FuseLLM-7B

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/Wanfq/FuseLLM-7B/resolve/main

set FILES[0]=README.md
set FILES[1]=config.json
set FILES[2]=generation_config.json
set FILES[3]=special_tokens_map.json
set FILES[4]=tokenizer.model
set FILES[5]=tokenizer_config.json
set FILES[6]=pytorch_model.bin.index.json
set FILES[7]=pytorch_model-00001-of-00002.bin
set FILES[8]=pytorch_model-00002-of-00002.bin

for /L %%i in (0,1,8) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
