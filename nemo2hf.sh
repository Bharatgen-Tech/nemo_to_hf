#!/bin/bash
SOURCE_NEMO_WEIGHT_FOLDER="/fsxnew/viraj.thakur/scripts/experiments/17B-cpt-MMLU/exp2/results/reduced_train_loss=0.59580-step=203-consumed_samples=391680.0-last/weights"
TARGET_HF_FOLDER="/fsxnew/viraj.thakur/scripts/experiments/17B-cpt-MMLU/exp2/results/hf-checkpoint"
TOKENIZER_PATH="/fsxnew/bhargav.patel/nemo_to_hf/tokenizer"
python3 /fsxnew/bhargav.patel/nemo_to_hf/nemo2hf.py \
        --checkpoint-path=$SOURCE_NEMO_WEIGHT_FOLDER \
        --target-path=$TARGET_HF_FOLDER \
        --force-bf16  \
        --override-tokenizer-path $TOKENIZER_PATH
cp /fsxnew/bhargav.patel/nemo_to_hf/configuration_bailing_moe_v2.py $TARGET_HF_FOLDER 
cp /fsxnew/bhargav.patel/nemo_to_hf/modeling_bailing_moe_v2.py $TARGET_HF_FOLDER  



