"""
Split data/raw/examples.jsonl into train/val sets and format each example
as an instruction-tuning record (instruction + expected JSON output).

Run:
    python data/split_dataset.py --in data/raw/examples.jsonl --val-fraction 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.schema import OUTPUT_SCHEMA_DESCRIPTION  # noqa: E402


def format_example(raw: dict) -> dict:
    """Convert a raw {message, label} record into an instruction-tuning
    {instruction, input, output} record, in the same shape Unsloth's
    training script expects (see training/train.py)."""
    return {
        "instruction": (
            "You classify a restaurant manager's message into a structured "
            "action for a POS system. " + OUTPUT_SCHEMA_DESCRIPTION
        ),
        "input": raw["message"],
        "output": json.dumps(raw["label"], ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=Path, default=Path("data/raw/examples.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lines = args.in_path.read_text(encoding="utf-8").splitlines()
    examples = [format_example(json.loads(line)) for line in lines if line.strip()]

    rng = random.Random(args.seed)
    rng.shuffle(examples)

    val_count = int(len(examples) * args.val_fraction)
    val_examples = examples[:val_count]
    train_examples = examples[val_count:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in [("train", train_examples), ("val", val_examples)]:
        out_path = args.out_dir / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for ex in subset:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{name}: {len(subset)} examples -> {out_path}")


if __name__ == "__main__":
    main()
