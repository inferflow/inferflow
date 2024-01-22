: Model: internlm-chat-20b
: InternLM-20B was pre-trained on over 2.3T Tokens containing high-quality English, Chinese, and code data.
: URL: https://huggingface.co/internlm/internlm-chat-20b

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/internlm/internlm-chat-20b/resolve/main

set FILES[0]=README.md
set FILES[1]=config.json
set FILES[2]=configuration_internlm.py
set FILES[3]=generation_config.json
set FILES[4]=modeling_internlm.py
set FILES[5]=special_tokens_map.json
set FILES[6]=tokenization_internlm.py
set FILES[7]=tokenizer.model
set FILES[8]=tokenizer_config.json
set FILES[9]=pytorch_model.bin.index.json
set FILES[10]=pytorch_model-00001-of-00006.bin
set FILES[11]=pytorch_model-00002-of-00006.bin
set FILES[12]=pytorch_model-00003-of-00006.bin
set FILES[13]=pytorch_model-00004-of-00006.bin
set FILES[14]=pytorch_model-00005-of-00006.bin
set FILES[15]=pytorch_model-00006-of-00006.bin

for /L %%i in (0,1,15) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
