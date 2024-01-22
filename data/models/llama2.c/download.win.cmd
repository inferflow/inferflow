: Model: tinyllamas
: A Llama 2 architecture model series trained on the TinyStories dataset, intended for use in the llama2.c project (https://github.com/karpathy/llama2.c).
: URL: https://huggingface.co/karpathy/tinyllamas

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/karpathy/tinyllamas/resolve/main

set FILES[0]=README.md
set FILES[1]=stories15M.bin
set FILES[2]=stories110M.bin

for /L %%i in (0,1,2) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
