#!/bin/bash
# Re-score an existing eval CSV (from run_eval.sh / generation/evaluate.py)
# with an independent AES verifier, for the cross-evaluator transfer check in §5.4 of the paper.
#
# Works both as a plain shell script and as a Slurm batch job:
#   INPUT_CSV=results/grpo_.../checkpoint-XXXX/eval_total_test.csv \
#   AES_MODEL=Heejindo/scorer_deberta_f0 \
#   bash scripts/run_rescore.sh
#
# Env vars:
#   INPUT_CSV   eval CSV produced by generation/evaluate.py (required)
#   AES_MODEL   HF model id / local path of the independent scorer (required)
#   HF_TOKEN    Hugging Face access token (only needed for private models)

#SBATCH --job-name=swim_rescore
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

: "${INPUT_CSV:?Set INPUT_CSV to an eval CSV produced by generation/evaluate.py}"
: "${AES_MODEL:?Set AES_MODEL to the independent scorer HF model id or local path}"

mkdir -p logs

python3 generation/rescore.py \
    --input_csv "$INPUT_CSV" \
    --aes_model "$AES_MODEL" \
    --aes_batch 16 \
    --max_src_len 512
