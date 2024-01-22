: Model: Yi-6B-200K
: The Yi series models are trained from strach by 01.AI. They follow the same model architecture as LLaMA.
: URL: https://huggingface.co/01-ai/Yi-6B-200K

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/01-ai/Yi-6B-200K/resolve/main

set FILES[0]=LICENSE
set FILES[1]=README.md
set FILES[2]=config.json
set FILES[3]=generation_config.json
set FILES[4]=tokenizer.json
set FILES[5]=tokenizer.model
set FILES[6]=tokenizer_config.json
set FILES[7]=model.safetensors.index.json
set FILES[8]=model-00001-of-00002.safetensors
set FILES[9]=model-00002-of-00002.safetensors

for /L %%i in (0,1,9) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
