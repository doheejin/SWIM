#!/bin/bash
# Rubric-grounded prompting baseline (CTP+FS / SRLP+FS) on a Qwen backbone.
#
# Works both as a plain shell script and as a Slurm batch job:
#   bash scripts/run_prompt_qwen.sh
#   sbatch --export=ALL,MODE=ctp_fs scripts/run_prompt_qwen.sh
#
# Env vars:
#   MODE        one of {ctp_fs, srlp_fs} (default: ctp_fs)
#   FOLD        ASAP fold index, 0-4 (default: 0)
#   BASE_MODEL  HF model id of the generator backbone (default: Qwen/Qwen2.5-7B-Instruct)
#   AES_MODEL   HF model id / local path of the frozen AES verifier (default: Heejindo/scorer_f<FOLD>,
#               i.e. the ArTS checkpoint trained on the SAME fold — each fold has its own verifier)
#   GEN_BATCH   generation batch size (default: 8)
#   HF_TOKEN    Hugging Face access token, required only if AES_MODEL/BASE_MODEL are private

#SBATCH --job-name=swim_prompt_qwen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
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

# --- Hugging Face auth (only needed for private models) ---
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

MODE="${MODE:-ctp_fs}"
FOLD="${FOLD:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
AES_MODEL="${AES_MODEL:-Heejindo/scorer_f${FOLD}}"
GEN_BATCH="${GEN_BATCH:-8}"

DATA_CSV="$PROJECT_DIR/dataset/asap/fold_${FOLD}/total_test.csv"
OUT_DIR="$PROJECT_DIR/results/prompt_qwen/fold_${FOLD}"
mkdir -p "$OUT_DIR" logs

echo "=========================================================="
echo "Prompting $BASE_MODEL with $MODE (fold $FOLD)"
echo "  data:      $DATA_CSV"
echo "  scorer:    $AES_MODEL"
echo "  gen_batch: $GEN_BATCH"
echo "=========================================================="

python3 generation/prompt_qwen.py \
    --mode "$MODE" \
    --base_model "$BASE_MODEL" \
    --data_csv "$DATA_CSV" \
    --aes_model "$AES_MODEL" \
    --output_dir "$OUT_DIR" \
    --gen_batch "$GEN_BATCH" \
    --aes_batch 16 \
    --max_new_tokens 1024 \
    --max_seq_len 12288 \
    --max_src_len 1024 \
    --bf16
