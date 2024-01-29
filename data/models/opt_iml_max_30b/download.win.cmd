: Model: OPT-IML-MAX-30B
: OPT-IML (OPT + Instruction Meta-Learning) is a set of instruction-tuned versions of OPT, on a collection of ~2000 NLP tasks gathered from 8 NLP benchmarks, called OPT-IML Bench. OPT-IML was trained on 1500 tasks with several tasks held-out for purposes of downstream evaluation. OPT-IML-Max was trained on all ~2000 tasks.
: URL: https://huggingface.co/facebook/opt-iml-max-30b

@echo off
setlocal enabledelayedexpansion

set SRC_DIR=https://huggingface.co/facebook/opt-iml-max-30b/resolve/main

set FILES[0]=LICENSE.md
set FILES[1]=README.md
set FILES[2]=config.json
set FILES[3]=generation_config.json
set FILES[4]=merges.txt
set FILES[5]=special_tokens_map.json
set FILES[6]=tokenizer_config.json
set FILES[7]=vocab.json
set FILES[8]=pytorch_model.bin.index.json
set FILES[9]=pytorch_model-00001-of-00007.bin
set FILES[10]=pytorch_model-00002-of-00007.bin
set FILES[11]=pytorch_model-00003-of-00007.bin
set FILES[12]=pytorch_model-00004-of-00007.bin
set FILES[13]=pytorch_model-00005-of-00007.bin
set FILES[14]=pytorch_model-00006-of-00007.bin
set FILES[15]=pytorch_model-00007-of-00007.bin

for /L %%i in (0,1,15) do (
    echo Downloading !FILES[%%i]! from %SRC_DIR%...
    call curl -L "%SRC_DIR%/!FILES[%%i]!" -o !FILES[%%i]!
)
