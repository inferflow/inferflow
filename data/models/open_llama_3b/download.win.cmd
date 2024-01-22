: Model: open_llama_3b
: A permissively licensed open source reproduction of Meta AI's LLaMA large language model.
: URL: https://huggingface.co/openlm-research/open_llama_3b

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/openlm-research/open_llama_3b/resolve/main

set FILES[0]=README.md
set FILES[1]=config.json
set FILES[2]=generation_config.json
set FILES[3]=special_tokens_map.json
set FILES[4]=tokenizer.model
set FILES[5]=tokenizer_config.json
set FILES[6]=pytorch_model.bin

for /L %%i in (0,1,6) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
