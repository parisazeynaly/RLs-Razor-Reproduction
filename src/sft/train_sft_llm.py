import argparse
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer


def build_prompt(question: str, choices: Dict[str, str]) -> str:
    """
    Unified MCQ prompt. The model must output ONLY the letter.
    """
    options = "\n".join([f"{k}. {v}" for k, v in choices.items()])
    return (
        "You are a helpful assistant.\n\n"
        "Answer the following multiple-choice question.\n"
        "Choose the correct option and respond with ONLY the letter (A, B, C, or D).\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options}\n\n"
        "Answer:"
    )


def format_sciq_example(ex: Dict[str, Any]) -> Dict[str, str]:
    """
    SciQ: correct_answer + 3 distractors.
    We will always place correct answer as D for deterministic labeling.
    """
    choices = {
        "A": ex["distractor1"],
        "B": ex["distractor2"],
        "C": ex["distractor3"],
        "D": ex["correct_answer"],
    }
    prompt = build_prompt(ex["question"], choices)
    # SFT target: append the correct letter (D)
    text = prompt + " D"
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    # Data controls (Colab-friendly)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_train_samples", type=int, default=4000)
    parser.add_argument("--max_eval_samples", type=int, default=500)

    # Training controls
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save run config for reproducibility
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Loading tokenizer/model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    # Llama Instruct usually has a pad token issue; set pad token to eos if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False  # important for training

    print("Loading SciQ dataset...")
    ds_train = load_dataset("allenai/sciq", split=args.split)
    ds_eval = load_dataset("allenai/sciq", split="validation")

    # Shuffle + subset for Colab time control
    ds_train = ds_train.shuffle(seed=args.seed)
    ds_eval = ds_eval.shuffle(seed=args.seed)

    if args.max_train_samples > 0:
        ds_train = ds_train.select(range(min(args.max_train_samples, len(ds_train))))
    if args.max_eval_samples > 0:
        ds_eval = ds_eval.select(range(min(args.max_eval_samples, len(ds_eval))))

    # Map to a single "text" field for SFTTrainer
    ds_train = ds_train.map(format_sciq_example, remove_columns=ds_train.column_names)
    ds_eval = ds_eval.map(format_sciq_example, remove_columns=ds_eval.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=200,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        warmup_steps=50,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=training_args,
        packing=False,
    )

    print("Starting SFT training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Done. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

