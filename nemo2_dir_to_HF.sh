#!/bin/bash
set -euo pipefail

# =========================
# CONFIGURATION
# =========================
CHECKPOINT_ROOT="/17B_production/results_backup_bhargav"
HF_ROOT="/bhargav/CHECKPOINTS/17B_aws_ckpt_$(date +%d_%m_%Y)"

TOKENIZER_PATH="/17B_production/14B_Architecture_Abalation/tokenizer"
CONFIG_FILE="/bhargav/NEMO2HF/configuration_bailing_moe_v2.py"
MODEL_FILE="/bhargav/NEMO2HF/modeling_bailing_moe_v2.py"

LOG_FILE="/bhargav/NEMO2HF/logs/nemo2hf_dir_conversion_$(date +%d_%m_%Y).log"

mkdir -p "$HF_ROOT"
echo "==== Conversion started at $(date) ====" | tee -a "$LOG_FILE"

# =========================
# LOOP OVER CHECKPOINTS
# =========================
for CKPT_DIR in "$CHECKPOINT_ROOT"/reduced_train_loss=*; do
    [ -d "$CKPT_DIR" ] || continue

    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "Processing: $CKPT_DIR" | tee -a "$LOG_FILE"

    # Extract step number
    STEP=$(echo "$CKPT_DIR" | sed -n 's/.*step=\([0-9]\+\).*/\1/p')

    if [[ -z "$STEP" ]]; then
        echo "⚠️  Could not extract step, skipping $CKPT_DIR" | tee -a "$LOG_FILE"
        continue
    fi

    SOURCE_NEMO_WEIGHT_FOLDER="$CKPT_DIR/weights"
    TARGET_HF_FOLDER="$HF_ROOT/${STEP}_HF"

    echo "Step: $STEP" | tee -a "$LOG_FILE"
    echo "Target HF folder: $TARGET_HF_FOLDER" | tee -a "$LOG_FILE"

    mkdir -p "$TARGET_HF_FOLDER"

    # =========================
    # RUN CONVERSION
    # =========================
    python3 nemo2hf.py \
        --checkpoint-path "$SOURCE_NEMO_WEIGHT_FOLDER" \
        --target-path "$TARGET_HF_FOLDER" \
        --force-bf16 \
        --override-tokenizer-path "$TOKENIZER_PATH" \
        >> "$LOG_FILE" 2>&1

    # =========================
    # COPY MODEL FILES
    # =========================
    cp "$CONFIG_FILE" "$TARGET_HF_FOLDER"
    cp "$MODEL_FILE" "$TARGET_HF_FOLDER"

    echo "✅ Finished step $STEP" | tee -a "$LOG_FILE"
done

echo "==== Conversion completed at $(date) ====" | tee -a "$LOG_FILE"
