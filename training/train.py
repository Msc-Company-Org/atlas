"""
Atlas — QLoRA fine-tuning for knowledge-grounded QA.

Reference training script for the MSC Labs Atlas model. Fine-tunes
Qwen2.5-7B-Instruct on grounded {context, question, answer} triples so the
model answers ONLY from the provided context, cites it, and abstains when the
answer is not present.

Two-stage protocol:
    Stage 1 (smoke test):  python train.py --config training/config.yaml --smoke_test
    Stage 2 (full run):    python train.py --config training/config.yaml

Illustrative reference code. Numbers in the repo are from the MSC Labs eval harness.
"""

import argparse

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

SYSTEM_PROMPT = (
    "You are a knowledge-base assistant. Answer ONLY from the provided context. "
    "Cite the supporting text. If the context does not contain the answer, say you "
    "don't know rather than guessing."
)


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA training for Atlas grounded QA.")
    p.add_argument("--config", default="training/config.yaml")
    p.add_argument("--smoke_test", action="store_true",
                   help="Stage 1: 1 epoch on a small subset to validate the pipeline.")
    return p.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def build_model(cfg):
    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_id"],
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"]
    )
    lc = cfg["lora"]
    lora = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        bias=lc["bias"],
        task_type=lc["task_type"],
        target_modules=lc["target_modules"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def format_example(tokenizer, data_cfg):
    """Render each row into the chat template the base model expects."""
    ctx_f, q_f, a_f = data_cfg["context_field"], data_cfg["question_field"], data_cfg["answer_field"]

    def _fmt(row):
        user = f"Context:\n{row[ctx_f]}\n\nQuestion: {row[q_f]}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": row[a_f]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    return _fmt


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg, train_cfg = cfg["data"], cfg["training"]

    tokenizer = build_tokenizer(cfg["base_model_id"])
    model = build_model(cfg)

    train_path = data_cfg["sample_file"] if args.smoke_test else data_cfg["train_file"]
    dataset = load_dataset("json", data_files={"train": train_path}, split="train")
    if args.smoke_test:
        n = min(cfg["smoke_test"]["max_train_rows"], len(dataset))
        dataset = dataset.select(range(n))
        train_cfg["num_train_epochs"] = cfg["smoke_test"]["num_train_epochs"]
        print(f"[smoke test] training on {n} rows for {train_cfg['num_train_epochs']} epoch")

    dataset = dataset.map(format_example(tokenizer, data_cfg),
                          remove_columns=dataset.column_names)

    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        optim=train_cfg["optim"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        max_seq_length=data_cfg["max_seq_len"],
        dataset_text_field="text",
        packing=False,
        seed=cfg["seed"],
        report_to="none",
    )

    trainer = SFTTrainer(model=model, args=sft_config,
                         train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()

    trainer.model.save_pretrained(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    print(f"Saved LoRA adapter to {train_cfg['output_dir']}")


if __name__ == "__main__":
    main()
