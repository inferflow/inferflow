: Model: falcon-40b-instruct
: Falcon-40B-Instruct is a 40B parameters causal decoder-only model built by TII based on Falcon-40B and finetuned on a mixture of Baize. It is made available under the Apache 2.0 license.
: URL: https://huggingface.co/tiiuae/falcon-40b-instruct

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/tiiuae/falcon-40b-instruct/resolve/main

set FILES[0]=README.md
set FILES[1]=config.json
set FILES[2]=configuration_falcon.py
set FILES[3]=generation_config.json
set FILES[4]=handler.py
set FILES[5]=modeling_falcon.py
set FILES[6]=special_tokens_map.json
set FILES[7]=tokenizer.json
set FILES[8]=tokenizer_config.json
set FILES[9]=pytorch_model.bin.index.json
set FILES[10]=pytorch_model-00001-of-00009.bin
set FILES[11]=pytorch_model-00002-of-00009.bin
set FILES[12]=pytorch_model-00003-of-00009.bin
set FILES[13]=pytorch_model-00004-of-00009.bin
set FILES[14]=pytorch_model-00005-of-00009.bin
set FILES[15]=pytorch_model-00006-of-00009.bin
set FILES[16]=pytorch_model-00007-of-00009.bin
set FILES[17]=pytorch_model-00008-of-00009.bin
set FILES[18]=pytorch_model-00009-of-00009.bin

for /L %%i in (0,1,18) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
