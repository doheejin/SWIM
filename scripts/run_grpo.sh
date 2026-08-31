#!/bin/bash
# Step 2: GRPO with the Proficiency Alignment Reward (PAR), initialized from an SFT checkpoint.
#
# Works both as a plain shell script and as a Slurm batch job:
#   SFT_CHECKPOINT=results/sft_qwen25-7b-instruct_f0/checkpoint-XXXX bash scripts/run_grpo.sh
#   sbatch --export=ALL,SFT_CHECKPOINT=... scripts/run_grpo.sh
#
# Env vars:
#   SFT_CHECKPOINT  path to the SFT LoRA checkpoint to start from (required)
#   MODEL           base HF model id, must match the SFT backbone (default: Qwen/Qwen2.5-7B-Instruct)
#   FOLD            ASAP fold index, 0-4 (default: 0)
#   AES_MODEL       HF model id / local path of the frozen AES verifier (default: Heejindo/scorer_f<FOLD>,
#                   i.e. the ArTS checkpoint trained on the SAME fold — each fold has its own verifier)
#   RUN_NAME        output directory name under results/ (default: grpo_<model-slug>_f<fold>)
#   HF_TOKEN        Hugging Face access token (only needed for private models)
#   WANDB_MODE      set to "disabled" to skip experiment tracking

#SBATCH --job-name=swim_grpo
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

: "${SFT_CHECKPOINT:?Set SFT_CHECKPOINT to the SFT LoRA checkpoint produced by run_sft.sh}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
FOLD="${FOLD:-0}"
AES_MODEL="${AES_MODEL:-Heejindo/scorer_f${FOLD}}"
MODEL_SLUG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
RUN_NAME="${RUN_NAME:-grpo_${MODEL_SLUG}_f${FOLD}}"
OUTPUT_DIR="$PROJECT_DIR/results/$RUN_NAME"
mkdir -p "$OUTPUT_DIR" logs

# NOTE: no repetition_penalty is applied here — it improves greedy fluency
# but distorts the AES reward distribution (the verifier was trained on
# natural essays where mild repetition is normal).
python3 rl/grpo_train.py \
    --sft_checkpoint "$SFT_CHECKPOINT" \
    --aes_model "$AES_MODEL" \
    --base_model "$MODEL" \
    --data_path "dataset/gen/not_norm/fold_${FOLD}" \
    --output_path "$OUTPUT_DIR" \
    --mode multitrait \
    --num_generations 4 \
    --epoch 1 \
    --lr 1e-5 \
    --train_batch 8 \
    --grad_accum 2 \
    --max_prompt_len 2500 \
    --max_completion_len 1024 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --bf16 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.25
