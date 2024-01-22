: Model: Phi-2
: Phi-2 is a Transformer with 2.7 billion parameters. It was trained using the same data sources as Phi-1.5, augmented with a new data source that consists of various NLP synthetic texts and filtered websites (for safety and educational value).
: URL: https://huggingface.co/microsoft/phi-2

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/microsoft/phi-2/resolve/main

set FILES[0]=CODE_OF_CONDUCT.md
set FILES[1]=LICENSE
set FILES[2]=NOTICE.md
set FILES[3]=README.md
set FILES[4]=SECURITY.md
set FILES[5]=added_tokens.json
set FILES[6]=config.json
set FILES[7]=configuration_phi.py
set FILES[8]=generation_config.json
set FILES[9]=merges.txt
set FILES[10]=modeling_phi.py
set FILES[11]=special_tokens_map.json
set FILES[12]=tokenizer.json
set FILES[13]=tokenizer_config.json
set FILES[14]=vocab.json
set FILES[15]=model.safetensors.index.json
set FILES[16]=model-00001-of-00002.safetensors
set FILES[17]=model-00002-of-00002.safetensors

for /L %%i in (0,1,17) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
