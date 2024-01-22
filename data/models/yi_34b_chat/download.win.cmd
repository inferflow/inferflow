: Model: Yi-34B-Chat
: The Yi series models are trained from strach by 01.AI. They follow the same model architecture as LLaMA.
: Data homepage: https://huggingface.co/01-ai/Yi-34B-Chat

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/01-ai/Yi-34B-Chat/resolve/main

set FILES[0]=LICENSE
set FILES[1]=README.md
set FILES[2]=config.json
set FILES[3]=generation_config.json
set FILES[4]=special_tokens_map.json
set FILES[5]=tokenizer.model
set FILES[6]=tokenizer_config.json
set FILES[7]=model.safetensors.index.json
set FILES[8]=model-00001-of-00015.safetensors
set FILES[9]=model-00002-of-00015.safetensors
set FILES[10]=model-00003-of-00015.safetensors
set FILES[11]=model-00004-of-00015.safetensors
set FILES[12]=model-00005-of-00015.safetensors
set FILES[13]=model-00006-of-00015.safetensors
set FILES[14]=model-00007-of-00015.safetensors
set FILES[15]=model-00008-of-00015.safetensors
set FILES[16]=model-00009-of-00015.safetensors
set FILES[17]=model-00010-of-00015.safetensors
set FILES[18]=model-00011-of-00015.safetensors
set FILES[19]=model-00012-of-00015.safetensors
set FILES[20]=model-00013-of-00015.safetensors
set FILES[21]=model-00014-of-00015.safetensors
set FILES[22]=model-00015-of-00015.safetensors

for /L %%i in (0,1,22) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
