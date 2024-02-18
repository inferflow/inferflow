: Model: Mixtral-8x7B-Instruct-v0.1
: The Mixtral-8x7B model is a pretrained generative Sparse Mixture of Experts. It outperforms Llama 2 70B on most benchmarks the authors tested.
: URL: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main

set FILES[0]=README.md
set FILES[1]=config.json
set FILES[2]=generation_config.json
set FILES[3]=special_tokens_map.json
set FILES[4]=tokenizer.json
set FILES[5]=tokenizer.model
set FILES[6]=tokenizer_config.json
set FILES[7]=model.safetensors.index.json
set FILES[8]=model-00001-of-00019.safetensors
set FILES[9]=model-00002-of-00019.safetensors
set FILES[10]=model-00003-of-00019.safetensors
set FILES[11]=model-00004-of-00019.safetensors
set FILES[12]=model-00005-of-00019.safetensors
set FILES[13]=model-00006-of-00019.safetensors
set FILES[14]=model-00007-of-00019.safetensors
set FILES[15]=model-00008-of-00019.safetensors
set FILES[16]=model-00009-of-00019.safetensors
set FILES[17]=model-00010-of-00019.safetensors
set FILES[18]=model-00011-of-00019.safetensors
set FILES[19]=model-00012-of-00019.safetensors
set FILES[20]=model-00013-of-00019.safetensors
set FILES[21]=model-00014-of-00019.safetensors
set FILES[22]=model-00015-of-00019.safetensors
set FILES[23]=model-00016-of-00019.safetensors
set FILES[24]=model-00017-of-00019.safetensors
set FILES[25]=model-00018-of-00019.safetensors
set FILES[26]=model-00019-of-00019.safetensors

for /L %%i in (0,1,26) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)

