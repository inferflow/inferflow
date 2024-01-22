: Model: ChatGLM2-6B
: ChatGLM2-6B is the second-generation version of the open-source bilingual (Chinese-English) chat model ChatGLM-6B.
: URL: https://huggingface.co/THUDM/chatglm2-6b

 @echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/THUDM/chatglm2-6b/resolve/main

set FILES[0]=MODEL_LICENSE
set FILES[1]=README.md
set FILES[2]=config.json
set FILES[3]=configuration_chatglm.py
set FILES[4]=modeling_chatglm.py
set FILES[5]=quantization.py
set FILES[6]=tokenization_chatglm.py
set FILES[7]=tokenizer.model
set FILES[8]=tokenizer_config.json
set FILES[9]=pytorch_model.bin.index.json
set FILES[10]=pytorch_model-00001-of-00007.bin
set FILES[11]=pytorch_model-00002-of-00007.bin
set FILES[12]=pytorch_model-00003-of-00007.bin
set FILES[13]=pytorch_model-00004-of-00007.bin
set FILES[14]=pytorch_model-00005-of-00007.bin
set FILES[15]=pytorch_model-00006-of-00007.bin
set FILES[16]=pytorch_model-00007-of-00007.bin

for /L %%i in (0,1,16) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
