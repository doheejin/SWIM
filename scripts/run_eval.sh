#!/bin/bash
# Evaluate an SFT or GRPO checkpoint: generate essays for the test split,
# score them with the frozen AES verifier, and report QWK.
#
# Works both as a plain shell script and as a Slurm batch job:
#   CHECKPOINT=results/sft_qwen25-7b-instruct_f0/checkpoint-XXXX bash scripts/run_eval.sh
#   sbatch --export=ALL,CHECKPOINT=... scripts/run_eval.sh
#
# Env vars:
#   CHECKPOINT  path to the LoRA checkpoint to evaluate (required)
#   MODEL       base HF model id, must match the checkpoint's backbone (default: Qwen/Qwen2.5-7B-Instruct)
#   FOLD        ASAP fold index, 0-4 (default: 0)
#   AES_MODEL   HF model id / local path of the frozen AES verifier (default: Heejindo/scorer_f<FOLD>,
#               i.e. the ArTS checkpoint trained on the SAME fold as the checkpoint being evaluated —
#               each fold has its own verifier, evaluate with a mismatched fold's verifier and you leak
#               training data into the "held-out" score)
#   USE_4BIT    set to "1" if the checkpoint was trained with --use_4bit
#   HF_TOKEN    Hugging Face access token (only needed for private models)

#SBATCH --job-name=swim_eval
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

if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

: "${CHECKPOINT:?Set CHECKPOINT to the SFT/GRPO checkpoint directory to evaluate}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
FOLD="${FOLD:-0}"
AES_MODEL="${AES_MODEL:-Heejindo/scorer_f${FOLD}}"
mkdir -p logs

EXTRA_ARGS=()
if [ "${USE_4BIT:-0}" = "1" ]; then
    EXTRA_ARGS+=(--use_4bit)
fi

python3 generation/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --base_model "$MODEL" \
    --aes_model "$AES_MODEL" \
    --data_path "dataset/gen/not_norm/fold_${FOLD}" \
    --data_file total_test.csv \
    --mode multitrait \
    --max_new_tokens 1024 \
    --max_src_len 1024 \
    --gen_batch 16 \
    --aes_batch 16 \
    --bf16 \
    "${EXTRA_ARGS[@]}"
