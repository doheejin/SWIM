"""
Basic SFT for essay generation.
"""

import os
import sys

HUB = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HOME"] = HUB
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HUB, "transformers"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(HUB, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HUB, "hub"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import wandb
import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from utils.data_utils import (
    load_sft_dataset,
    load_prompt_map,
    format_for_sft,
    CompletionOnlyCollator,
    get_response_template_ids,
)

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "swim")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")  # None -> use the wandb account's default entity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",   type=str, default="dataset/gen/not_norm/fold_0")
    p.add_argument("--prompt_jsonl", type=str, default="dataset/gen/prompt.jsonl")
    p.add_argument("--output_path", type=str, default="results/sft_qwen_f0")
    p.add_argument("--model",       type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--mode",        type=str, default="multitrait", choices=["multitrait", "holistic"])

    # Training
    p.add_argument("--epoch",       type=int,   default=3)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--train_batch", type=int,   default=1)
    p.add_argument("--valid_batch", type=int,   default=1)
    p.add_argument("--grad_accum",  type=int,   default=16)
    p.add_argument("--max_seq_len", type=int,   default=1024)
    p.add_argument("--eval_steps",  type=int,   default=500)
    p.add_argument("--save_steps",  type=int,   default=500)

    # LoRA
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Precision
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--bf16",     action="store_true")
    p.add_argument("--fp16",     action="store_true")

    # Resume
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to checkpoint directory to resume training from")
    return p.parse_args()


def load_model(args):
    if args.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, cache_dir=HUB, quantization_config=bnb,
            device_map="auto", trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, cache_dir=HUB, device_map="auto",
            torch_dtype=dtype, trust_remote_code=True,
        )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        model.get_input_embeddings().weight.requires_grad_(True)
    return model


def main():
    args = parse_args()

    run_name = os.path.basename(args.output_path.rstrip("/"))

    if args.resume_from_checkpoint:
        ckpt_parent = os.path.dirname(args.resume_from_checkpoint.rstrip("/"))
        run_id_file = os.path.join(ckpt_parent, "wandb_run_id.txt")
        if os.path.exists(run_id_file):
            with open(run_id_file) as f:
                run_id = f.read().strip()
            wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, id=run_id, name=run_name, resume="must")
        else:
            run_id = wandb.util.generate_id()
            wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, id=run_id, name=run_name, resume="never")
    else:
        run_id = wandb.util.generate_id()
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, id=run_id, name=run_name, resume="never")
        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, "wandb_run_id.txt"), "w") as f:
            f.write(run_id)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(
        args.model, cache_dir=HUB, use_fast=True, trust_remote_code=True,
    )
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Datasets
    prompt_map = load_prompt_map(args.prompt_jsonl)
    train_ds, dev_ds = load_sft_dataset(args.data_path)
    mode = args.mode
    train_ds = train_ds.map(lambda ex: format_for_sft(tok, ex, mode, prompt_map))
    dev_ds   = dev_ds.map(  lambda ex: format_for_sft(tok, ex, mode, prompt_map))

    # Collator (completion-only loss)
    tmpl_ids = get_response_template_ids(tok, args.model)
    collator = CompletionOnlyCollator(
        tokenizer=tok,
        response_template_ids=tmpl_ids,
        max_length=args.max_seq_len,
    )

    # Model
    model = load_model(args)

    # LoRA
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # Training config (SFTConfig = new TRL API)
    sft_args = SFTConfig(
        output_dir=args.output_path,
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.valid_batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        dataloader_num_workers=1,
        report_to=["wandb"],
        run_name=run_name,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        max_length=args.max_seq_len,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        peft_config=lora_cfg,
    )
    trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.evaluate()
    trainer.save_model(args.output_path)
    wandb.finish()


if __name__ == "__main__":
    main()
