: Model: XVERSE-13B-Chat
: XVERSE-13B-Chat is the aligned version of model XVERSE-13B. XVERSE-13B is a multilingual large language model, independently developed by Shenzhen Yuanxiang Technology.
: URL: https://huggingface.co/xverse/XVERSE-13B-Chat

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/xverse/XVERSE-13B-Chat/resolve/main

set FILES[0]=MODEL_LICENSE.pdf
set FILES[1]=README.md
set FILES[2]=config.json
set FILES[3]=configuration_xverse.py
set FILES[4]=generation_config.json
set FILES[5]=modeling_xverse.py
set FILES[6]=quantization.py
set FILES[7]=special_tokens_map.json
set FILES[8]=tokenizer.json
set FILES[9]=tokenizer_config.json
set FILES[10]=pytorch_model.bin.index.json
set FILES[11]=pytorch_model-00001-of-00010.bin
set FILES[12]=pytorch_model-00002-of-00010.bin
set FILES[13]=pytorch_model-00003-of-00010.bin
set FILES[14]=pytorch_model-00004-of-00010.bin
set FILES[15]=pytorch_model-00005-of-00010.bin
set FILES[16]=pytorch_model-00006-of-00010.bin
set FILES[17]=pytorch_model-00007-of-00010.bin
set FILES[18]=pytorch_model-00008-of-00010.bin
set FILES[19]=pytorch_model-00009-of-00010.bin
set FILES[20]=pytorch_model-00010-of-00010.bin

for /L %%i in (0,1,20) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
