"""
LoRA fine-tune a small open model on the SmartServe intent-extraction task.

Uses Unsloth for speed and lower VRAM usage on a single local GPU. Start
with the 1.5B model to get the full pipeline working end to end quickly —
data loading, prompt formatting, training loop, checkpoint saving — before
spending time on a bigger model. Swap MODEL_NAME to try Qwen2.5-7B-Instruct
once you have headroom and a working baseline to compare against.

Run:
    python training/train.py --config training/config.yaml

Requires a CUDA GPU. See training/requirements.txt for the environment —
keep this in its own venv, separate from SmartServe's Python services.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def format_prompts(example: dict) -> dict:
    return {
        "text": PROMPT_TEMPLATE.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    # Imported here, not at module level, so `--help` and config parsing
    # work even before the (heavy, GPU-only) unsloth/torch stack is set up.
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"Loading base model: {cfg['model_name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,  # auto-detect
        load_in_4bit=cfg["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["train_path"],
            "validation": cfg["val_path"],
        },
    )
    dataset = dataset.map(format_prompts)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        args=TrainingArguments(
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            warmup_steps=cfg["warmup_steps"],
            num_train_epochs=cfg["num_epochs"],
            learning_rate=cfg["learning_rate"],
            fp16=not cfg["load_in_4bit"],
            bf16=cfg["load_in_4bit"],
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=cfg["eval_steps"],
            save_strategy="steps",
            save_steps=cfg["eval_steps"],
            output_dir=cfg["output_dir"],
            seed=cfg["seed"],
            report_to="none",
        ),
    )

    print("Starting training...")
    trainer.train()

    final_dir = Path(cfg["output_dir"]) / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved fine-tuned adapter to {final_dir}")
    print("Next: python inference/quantize.py to export a GGUF for serving.")


if __name__ == "__main__":
    main()
