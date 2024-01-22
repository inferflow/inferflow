: Model: Mistral-7B-Instruct-v0.2
: The Mistral-7B-Instruct-v0.2 model is a instruct fine-tuned version of the Mistral-7B-v0.1 generative text model using a variety of publicly available conversation datasets. Mistral-7B-v0.1 is a transformer model, with the following architecture choices: grouped-query attention, sliding-window attention, byte-fallback bpe tokenizer.
: URL: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main

set FILES[0]=README.md
set FILES[1]=config.json
set FILES[2]=generation_config.json
set FILES[3]=special_tokens_map.json
set FILES[4]=tokenizer.json
set FILES[5]=tokenizer.model
set FILES[6]=tokenizer_config.json
set FILES[7]=model.safetensors.index.json
set FILES[8]=model-00001-of-00003.safetensors
set FILES[9]=model-00002-of-00003.safetensors
set FILES[10]=model-00003-of-00003.safetensors

for /L %%i in (0,1,10) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
