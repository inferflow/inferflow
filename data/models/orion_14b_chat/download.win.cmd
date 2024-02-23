: Model: Orion-14B-Chat
: Orion-14B series models are open-source multilingual large language models trained by OrionStarAI.
: URL: https://huggingface.co/OrionStarAI/Orion-14B-Chat

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/OrionStarAI/Orion-14B-Chat/resolve/main

set FILES[0]=LICENSE
set FILES[1]=ModelsCommunityLicenseAgreement
set FILES[2]=README.md
set FILES[3]=config.json
set FILES[4]=configuration_orion.py
set FILES[5]=generation_config.json
set FILES[6]=generation_utils.py
set FILES[7]=modeling_orion.py
set FILES[8]=special_tokens_map.json
set FILES[9]=tokenization_orion.py
set FILES[10]=tokenizer.model
set FILES[11]=tokenizer_config.json
set FILES[12]=pytorch_model.bin.index.json
set FILES[13]=pytorch_model-00001-of-00003.bin
set FILES[14]=pytorch_model-00002-of-00003.bin
set FILES[15]=pytorch_model-00003-of-00003.bin

for /L %%i in (0,1,15) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
