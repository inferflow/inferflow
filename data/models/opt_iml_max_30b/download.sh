# Model: OPT-IML-MAX-30B
# OPT-IML (OPT + Instruction Meta-Learning) is a set of instruction-tuned versions of OPT, on a collection of ~2000 NLP tasks gathered from 8 NLP benchmarks, called OPT-IML Bench. OPT-IML was trained on 1500 tasks with several tasks held-out for purposes of downstream evaluation. OPT-IML-Max was trained on all ~2000 tasks.
# URL: https://huggingface.co/facebook/opt-iml-max-30b

SRC_DIR=https://huggingface.co/facebook/opt-iml-max-30b/resolve/main

FILE[0]=LICENSE.md
FILE[1]=README.md
FILE[2]=config.json
FILE[3]=generation_config.json
FILE[4]=merges.txt
FILE[5]=special_tokens_map.json
FILE[6]=tokenizer_config.json
FILE[7]=vocab.json
FILE[8]=pytorch_model.bin.index.json
FILE[9]=pytorch_model-00001-of-00007.bin
FILE[10]=pytorch_model-00002-of-00007.bin
FILE[11]=pytorch_model-00003-of-00007.bin
FILE[12]=pytorch_model-00004-of-00007.bin
FILE[13]=pytorch_model-00005-of-00007.bin
FILE[14]=pytorch_model-00006-of-00007.bin
FILE[15]=pytorch_model-00007-of-00007.bin

for i in {0..15}
do
  wget ${SRC_DIR}/${FILE[$i]}
done
