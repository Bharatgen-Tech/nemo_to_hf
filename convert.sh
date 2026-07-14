#!/bin/bash
SOURCE_NEMO_WEIGHT_FOLDER="/fsxtmp/ajay.nagpal/PT-3-Training-aux_loss-1e-3/base-ckpt/reduced_train_loss=1.46411-step=1700014-consumed_samples=3481630720.0-last/weights"
TARGET_HF_FOLDER="/fsxnew/checkpoints-pretraining/17B_PT3/17B_PT3_1700k_FP32_v4"
TOKENIZER_PATH="/fsxnew/bhargav.patel/nemo_to_hf/tokenizer"
python3 /fsxnew/bhargav.patel/nemo_to_hf/nemo2hf_fp32.py \
        --checkpoint-path=$SOURCE_NEMO_WEIGHT_FOLDER \
        --target-path=$TARGET_HF_FOLDER \
        --force-bf16  \
        --override-tokenizer-path $TOKENIZER_PATH
cp /fsxnew/bhargav.patel/nemo_to_hf/configuration_bailing_moe_v2.py $TARGET_HF_FOLDER 
cp /fsxnew/bhargav.patel/nemo_to_hf/modeling_bailing_moe_v2.py $TARGET_HF_FOLDER  
