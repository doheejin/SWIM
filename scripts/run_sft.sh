#!/bin/bash
# Step 1: Supervised fine-tuning (SFT) on real (essay, trait-score) pairs.
#
# Works both as a plain shell script and as a Slurm batch job:
#   bash scripts/run_sft.sh
#   sbatch --export=ALL,MODEL=Qwen/Qwen3-4B,FOLD=0 scripts/run_sft.sh
#
# Env vars:
#   MODEL       HF model id of the backbone (default: Qwen/Qwen2.5-7B-Instruct)
#   FOLD        ASAP fold index, 0-4 (default: 0)
#   RUN_NAME    output directory name under results/ (default: sft_<model-slug>_f<fold>)
#   HF_TOKEN    Hugging Face access token (only needed for private models)
#   WANDB_MODE  set to "disabled" to skip experiment tracking

#SBATCH --job-name=swim_sft
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# Adjust the SBATCH directives above (partition, account, gres syntax, etc.)
# to match your cluster; they are ignored when this script is run with `bash`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- activate your environment here, e.g.: ---
# module load cuda/12.4
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate swim

export PYTHONPATH="$PROJECT_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
export WANDB_MODE="${WANDB_MODE:-online}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
FOLD="${FOLD:-0}"
MODEL_SLUG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
RUN_NAME="${RUN_NAME:-sft_${MODEL_SLUG}_f${FOLD}}"
OUTPUT_DIR="$PROJECT_DIR/results/$RUN_NAME"
mkdir -p "$OUTPUT_DIR" logs

python3 generation/sft_train.py \
    --data_path "dataset/gen/not_norm/fold_${FOLD}" \
    --output_path "$OUTPUT_DIR" \
    --model "$MODEL" \
    --mode multitrait \
    --epoch 5 \
    --lr 1e-4 \
    --train_batch 4 \
    --grad_accum 2 \
    --max_seq_len 2500 \
    --eval_steps 500 \
    --save_steps 500 \
    --lora_r 16 \
    --lora_alpha 32 \
    --use_4bit \
    --bf16
