: Model: Baichuan2-7B-Chat
: Baichuan 2 is the new generation of large-scale open-source language models launched by Baichuan Intelligence inc.. It is trained on a high-quality corpus with 2.6 trillion tokens and has achieved the best performance in authoritative Chinese and English benchmarks of the same size.
: URL: https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main

set FILES[0]=Baichuan2%20%E6%A8%A1%E5%9E%8B%E7%A4%BE%E5%8C%BA%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf
set FILES[1]=Community%20License%20for%20Baichuan2%20Model.pdf
set FILES[2]=README.md
set FILES[3]=config.json
set FILES[4]=configuration_baichuan.py
set FILES[5]=generation_config.json
set FILES[6]=generation_utils.py
set FILES[7]=modeling_baichuan.py
set FILES[8]=quantizer.py
set FILES[9]=special_tokens_map.json
set FILES[10]=tokenization_baichuan.py
set FILES[11]=tokenizer.model
set FILES[12]=tokenizer_config.json
set FILES[13]=pytorch_model.bin

for /L %%i in (0,1,13) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
