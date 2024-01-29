: Model: OPT-350M
: OPT was first introduced in "OPT: Open Pre-trained Transformer Language Models" (https://arxiv.org/abs/2205.01068) and first released in metaseq's repository on May 3rd 2022 by Meta AI. OPT was predominantly pretrained with English text, but a small amount of non-English data is still present within the training corpus via CommonCrawl. The model was pretrained using a causal language modeling (CLM) objective. OPT belongs to the same family of decoder-only models like GPT-3.
: URL: https://huggingface.co/facebook/opt-350m

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/facebook/opt-350m/resolve/main

set FILES[0]=LICENSE.md
set FILES[1]=README.md
set FILES[2]=config.json
set FILES[3]=generation_config.json
set FILES[4]=merges.txt
set FILES[5]=special_tokens_map.json
set FILES[6]=tokenizer_config.json
set FILES[7]=vocab.json
set FILES[8]=pytorch_model.bin

for /L %%i in (0,1,8) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
