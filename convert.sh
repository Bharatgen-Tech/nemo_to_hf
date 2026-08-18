#!/bin/bash
SOURCE_NEMO_WEIGHT_FOLDER="/fsx2/checkpoints-pretraining/pt3/pt3_500k_nemo/weights"
TARGET_HF_FOLDER="/fsx2/post-training-data/vijay/sft_scripts/sft_scripts/Models/param-17b/32k"
TOKENIZER_PATH="/fsx2/bhargav.patel/nemo_to_hf/tokenizer"
python3 /fsx2/bhargav.patel/nemo_to_hf/nemo2hf_fp32.py \
        --checkpoint-path=$SOURCE_NEMO_WEIGHT_FOLDER \
        --target-path=$TARGET_HF_FOLDER \
        --force-bf16  \
        --override-tokenizer-path $TOKENIZER_PATH
cp /fsx2/bhargav.patel/nemo_to_hf/configuration_bailing_moe_v2.py $TARGET_HF_FOLDER 
cp /fsx2/bhargav.patel/nemo_to_hf/modeling_bailing_moe_v2.py $TARGET_HF_FOLDER  
