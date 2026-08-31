"""
GRPO (Group Relative Policy Optimization) with AES as reward.
Requires: trl >= 0.15 (GRPOTrainer)
"""

import os
import sys
import json

HUB = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HOME"] = HUB
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HUB, "transformers"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(HUB, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HUB, "hub"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import wandb
import torch
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig

from aes.model import ArTSScorer
from utils.data_utils import load_grpo_dataset, load_grpo_eval_dataset, load_prompt_map, format_prompt_only

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "swim")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")  # None -> use the wandb account's default entity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_checkpoint", type=str, required=True,
                   help="SFT LoRA checkpoint to use as the initial policy.")
    p.add_argument("--aes_model",      type=str,
                   default="Heejindo/scorer_f0",
                   help="Must be the verifier trained on the SAME fold as --data_path/--sft_checkpoint, "
                        "e.g. Heejindo/scorer_f1 for fold_1. Each fold has its own verifier; using a "
                        "mismatched fold leaks that fold's training essays into the reward.")
    p.add_argument("--data_path",      type=str, default="dataset/gen/not_norm/fold_0")
    p.add_argument("--prompt_jsonl",   type=str, default="dataset/gen/prompt.jsonl")
    p.add_argument("--output_path",    type=str, default="results/grpo_f0")
    p.add_argument("--base_model",     type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--mode",           type=str, default="multitrait",
                   choices=["multitrait", "holistic"])

    # GRPO hyperparams
    p.add_argument("--num_generations", type=int,   default=4,
                   help="G: number of essays sampled per prompt.")
    p.add_argument("--epoch",           type=int,   default=1)
    p.add_argument("--lr",              type=float, default=5e-6)
    p.add_argument("--train_batch",     type=int,   default=1)
    p.add_argument("--grad_accum",      type=int,   default=16)
    p.add_argument("--max_prompt_len",  type=int,   default=512)
    p.add_argument("--max_completion_len", type=int, default=512)
    p.add_argument("--eval_steps",      type=int,   default=200)
    p.add_argument("--save_steps",      type=int,   default=200)
    p.add_argument("--enable_eval",     action="store_true",
                   help="Run eval during training. Off by default — eval is expensive in GRPO.")

    # LoRA (applied on top of SFT checkpoint; keep same rank)
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Precision
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--bf16",     action="store_true")
    p.add_argument("--fp16",     action="store_true")

    # vLLM rollout acceleration (TRL >= 0.16)
    p.add_argument("--use_vllm", action="store_true",
                   help="Use vLLM for rollout generation (much faster).")
    p.add_argument("--vllm_mode", type=str, default="colocate",
                   choices=["colocate", "server"],
                   help="colocate: vLLM shares the training GPU; server: separate process.")
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3,
                   help="Fraction of GPU memory reserved for vLLM (colocate mode).")
    return p.parse_args()


def build_reward_fn(scorer: ArTSScorer):
    """
    Returns a reward function compatible with TRL's GRPOTrainer.

    The dataset must include 'prompt_id' and 'target_json' columns
    (added by format_prompt_only).  GRPOTrainer passes them as kwargs.

    Args (called by GRPOTrainer):
        completions : list[str]  — G generated essays per prompt (flattened)
        prompt_id   : list[int]  — repeated G times per original row
        target_json : list[str]  — JSON-encoded target trait dicts
    Returns:
        list[float] — reward per completion
    """
    def reward_fn(completions, prompts=None, prompt_id=None, target_json=None, **kwargs):
        if prompt_id is None or target_json is None:
            return [0.0] * len(completions)

        targets = [json.loads(tj) for tj in target_json]
        pids    = [int(pid) for pid in prompt_id]

        # Batch AES scoring
        preds = scorer.score(completions, pids)

        return [
            scorer.reward(pred, target, pid)
            for pred, target, pid in zip(preds, targets, pids)
        ]

    return reward_fn


def main():
    args = parse_args()

    run_id = wandb.util.generate_id()
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, id=run_id, resume=True)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(
        args.sft_checkpoint, cache_dir=HUB, use_fast=True, trust_remote_code=True,
    )
    tok.padding_side = "left"   # left-padding for generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Dataset — prompt-only, with prompt_id + target_json columns
    prompt_map = load_prompt_map(args.prompt_jsonl)
    mode = args.mode

    train_raw = load_grpo_dataset(args.data_path)
    train_ds = train_raw.map(
        lambda ex: format_prompt_only(tok, ex, mode, prompt_map),
        remove_columns=train_raw.column_names,
    )

    if args.enable_eval:
        eval_raw = load_grpo_eval_dataset(args.data_path)
        eval_ds = eval_raw.map(
            lambda ex: format_prompt_only(tok, ex, mode, prompt_map),
            remove_columns=eval_raw.column_names,
        )
    else:
        eval_ds = None

    # AES reward model (frozen, CPU or separate GPU)
    scorer = ArTSScorer(args.aes_model, device="cuda", max_src_len=1024)
    reward_fn = build_reward_fn(scorer)

    # Policy model: base + SFT adapter (trainable)
    if args.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, cache_dir=HUB, quantization_config=bnb,
            device_map="auto", trust_remote_code=True,
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:
        dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, cache_dir=HUB, device_map="auto",
            torch_dtype=dtype, trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(base, args.sft_checkpoint, is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # GRPO config
    grpo_args = GRPOConfig(
        output_dir=args.output_path,
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        logging_steps=10,
        eval_strategy="steps" if args.enable_eval else "no",
        eval_steps=args.eval_steps if args.enable_eval else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        report_to=["wandb"],
        run_name=run_id,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        # GRPO-specific (TRL 1.x: no max_prompt_length; use vllm_max_model_length)
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_len,
        # NOTE: repetition_penalty was tried at 1.15 (commit on 2026-05-18) but
        # caused GRPO performance to drop sharply on Qwen2.5-7B (mean QWK
        # 0.664 -> 0.526, esp. -0.43 on narrative prompt 3). AES T5 was trained
        # on natural student essays where mild repetition is normal; rep_penalty
        # generations fall out of AES's distribution, producing noisy reward.
        # Default (1.0) restored.
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=(args.max_prompt_len + args.max_completion_len)
                              if args.use_vllm else None,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
    )
    # Auto-resume from the latest checkpoint in output_dir if any.
    resume = None
    if os.path.isdir(args.output_path):
        ckpts = [d for d in os.listdir(args.output_path) if d.startswith("checkpoint-")]
        if ckpts:
            resume = True  # Trainer picks the latest checkpoint-N

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_path)
    wandb.finish()


if __name__ == "__main__":
    main()
