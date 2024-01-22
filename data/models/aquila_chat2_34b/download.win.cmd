: Model: AquilaChat2-34B
: A chat model with 34-billion parameters in the Aquila2 series open sourced by BAAI.
: URL: https://huggingface.co/BAAI/AquilaChat2-34B

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/BAAI/AquilaChat2-34B/resolve/main

set FILES[0]=BAAI-Aquila-Model-License%20-Agreement.pdf
set FILES[1]=LICENSE
set FILES[2]=README.md
set FILES[3]=README_zh.md
set FILES[4]=added_tokens.json
set FILES[5]=config.json
set FILES[6]=configuration_aquila.py
set FILES[7]=generation_config.json
set FILES[8]=merges.txt
set FILES[9]=modeling_aquila.py
set FILES[10]=predict.py
set FILES[11]=special_tokens_map.json
set FILES[12]=tokenizer.json
set FILES[13]=tokenizer_config.json
set FILES[14]=vocab.json
set FILES[15]=pytorch_model.bin.index.json
set FILES[16]=pytorch_model-00001-of-00007.bin
set FILES[17]=pytorch_model-00002-of-00007.bin
set FILES[18]=pytorch_model-00003-of-00007.bin
set FILES[19]=pytorch_model-00004-of-00007.bin
set FILES[20]=pytorch_model-00005-of-00007.bin
set FILES[21]=pytorch_model-00006-of-00007.bin
set FILES[22]=pytorch_model-00007-of-00007.bin

for /L %%i in (0,1,22) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
