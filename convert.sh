#!/bin/bash
SOURCE_NEMO_WEIGHT_FOLDER="/fsxnew/checkpoints-pretraining/17B_PT2/17B_PT2_1M_decay_1_1_989k_nemo/weights"
TARGET_HF_FOLDER="/fsxnew/checkpoints-pretraining/17B_PT2/17B_PT2_1M_decay_1_1_989k_HF"
TOKENIZER_PATH="/fsxnew/bhargav.patel/nemo_to_hf/tokenizer"
python3 /fsxnew/bhargav.patel/nemo_to_hf/nemo2hf.py \
        --checkpoint-path=$SOURCE_NEMO_WEIGHT_FOLDER \
        --target-path=$TARGET_HF_FOLDER \
        --force-bf16  \
        --override-tokenizer-path $TOKENIZER_PATH
cp /fsxnew/bhargav.patel/nemo_to_hf/configuration_param2moe.py $TARGET_HF_FOLDER 
cp /fsxnew/bhargav.patel/nemo_to_hf/modeling_param2moe.py $TARGET_HF_FOLDER  
