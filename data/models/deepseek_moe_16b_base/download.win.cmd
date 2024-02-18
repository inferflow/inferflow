: Model: deepseek-moe-16b-base
: DeepSeekMoE 16B is a Mixture-of-Experts (MoE) language model with 16.4B parameters. It is trained from scratch on 2T English and Chinese tokens, and exhibits comparable performance with DeekSeek 7B and LLaMA2 7B, with only about 40% of computations.
: URL: https://huggingface.co/deepseek-ai/deepseek-moe-16b-base

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/resolve/main

set FILES[0]=README.md
set FILES[1]=config.json
set FILES[2]=configuration_deepseek.py
set FILES[3]=generation_config.json
set FILES[4]=modeling_deepseek.py
set FILES[5]=tokenizer.json
set FILES[6]=tokenizer_config.json
set FILES[7]=model.safetensors.index.json
set FILES[8]=model-00001-of-00007.safetensors
set FILES[9]=model-00002-of-00007.safetensors
set FILES[10]=model-00003-of-00007.safetensors
set FILES[11]=model-00004-of-00007.safetensors
set FILES[12]=model-00005-of-00007.safetensors
set FILES[13]=model-00006-of-00007.safetensors
set FILES[14]=model-00007-of-00007.safetensors

for /L %%i in (0,1,14) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
