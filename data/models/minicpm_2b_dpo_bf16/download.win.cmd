: Model: MiniCPM-2B-dpo-bf16
: MiniCPM is an End-Size LLM developed by ModelBest Inc. and TsinghuaNLP, with only 2.4B parameters excluding embeddings.
: URL: https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16/resolve/main

set FILES[0]=README.md
set FILES[1]=config.json
set FILES[2]=configuration_minicpm.py
set FILES[3]=generation_config.json
set FILES[4]=modeling_minicpm.py
set FILES[5]=special_tokens_map.json
set FILES[6]=tokenizer.json
set FILES[7]=tokenizer.model
set FILES[8]=tokenizer_config.json
set FILES[9]=pytorch_model.bin

for /L %%i in (0,1,9) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
