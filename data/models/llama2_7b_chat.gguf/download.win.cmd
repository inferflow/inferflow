: Model: Llama-2-7B-Chat-GGUF
: GGUF format model for Meta Llama 2's Llama 2 7B Chat (https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
: URL: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main

set FILES[0]=LICENSE.txt
set FILES[1]=Notice
set FILES[2]=README.md
set FILES[3]=USE_POLICY.md
set FILES[4]=config.json
set FILES[5]=llama-2-7b-chat.Q8_0.gguf

for /L %%i in (0,1,5) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
