"""
Evaluate the fine-tuned local model on menus it has never seen, and
compare against Claude Haiku as a baseline. This comparison is the core
result for your project writeup — it's what proves the model generalizes
rather than memorizes, and it's what tells you whether the local tier is
actually worth deploying in front of Haiku.

Run:
    python evaluation/evaluate.py \\
        --menus data/raw/menus.jsonl \\
        --model-path training/checkpoints/final \\
        --holdout-fraction 0.15

Produces evaluation/results/report.json and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.schema import INTENTS, OUTPUT_SCHEMA_DESCRIPTION  # noqa: E402
from data.generate_dataset import build_user_prompt, sample_slot_values  # noqa: E402

import anthropic


def build_test_set(menus: list[dict], n_per_intent: int, seed: int) -> list[dict]:
    """Generate held-out test examples the same way training data was
    generated, but against menus the model never trained on."""
    import random

    rng = random.Random(seed)
    client = anthropic.Anthropic()
    test_set = []
    for menu in menus:
        for intent_name in INTENTS:
            for _ in range(n_per_intent):
                slot_values = sample_slot_values(menu, intent_name, rng) if intent_name != "unclear" else {}
                if intent_name == "unclear":
                    message = "not sure what to do with the new supplier invoice"
                else:
                    prompt = build_user_prompt(menu, intent_name, slot_values)
                    resp = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=200,
                        system="Generate a natural manager message for this request. Output only the message.",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    message = resp.content[0].text.strip()
                test_set.append(
                    {
                        "menu": menu,
                        "message": message,
                        "expected_intent": intent_name,
                        "expected_slots": slot_values,
                    }
                )
    return test_set


def classify_with_haiku(client: anthropic.Anthropic, message: str) -> dict:
    system = "Classify this restaurant manager message. " + OUTPUT_SCHEMA_DESCRIPTION
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=system,
        messages=[{"role": "user", "content": message}],
    )
    try:
        return json.loads(resp.content[0].text.strip())
    except json.JSONDecodeError:
        return {"intent": "unclear", "confidence": 0.0, "slots": {}}


def classify_with_local_model(model, tokenizer, message: str) -> tuple[dict, float]:
    from training.train import PROMPT_TEMPLATE  # reuse the exact training format

    prompt = PROMPT_TEMPLATE.format(
        instruction="You classify a restaurant manager's message into a structured action for a POS system. "
        + OUTPUT_SCHEMA_DESCRIPTION,
        input=message,
        output="",
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    latency_ms = (time.time() - start) * 1000
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        return json.loads(text.strip()), latency_ms
    except json.JSONDecodeError:
        return {"intent": "unclear", "confidence": 0.0, "slots": {}}, latency_ms


def score(predicted: dict, expected_intent: str) -> bool:
    return predicted.get("intent") == expected_intent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--menus", type=Path, default=Path("data/raw/menus.jsonl"))
    parser.add_argument("--model-path", type=Path, default=Path("training/checkpoints/final"))
    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    parser.add_argument("--n-per-intent", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)  # different from data-gen seed
    parser.add_argument("--skip-local", action="store_true", help="Score Haiku only (no GPU needed)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first.")
        sys.exit(1)

    all_menus = [json.loads(line) for line in args.menus.read_text(encoding="utf-8").splitlines()]
    holdout_count = max(1, int(len(all_menus) * args.holdout_fraction))
    holdout_menus = all_menus[-holdout_count:]
    print(f"Evaluating against {len(holdout_menus)} menus never seen during training")

    test_set = build_test_set(holdout_menus, args.n_per_intent, args.seed)
    print(f"Built {len(test_set)} held-out test examples")

    haiku_client = anthropic.Anthropic()

    results = {"haiku": {"correct": 0, "total": 0}, "local": {"correct": 0, "total": 0, "latency_ms": []}}

    local_model, local_tokenizer = None, None
    if not args.skip_local:
        from unsloth import FastLanguageModel

        local_model, local_tokenizer = FastLanguageModel.from_pretrained(str(args.model_path))
        FastLanguageModel.for_inference(local_model)

    per_example = []
    for ex in test_set:
        haiku_pred = classify_with_haiku(haiku_client, ex["message"])
        haiku_correct = score(haiku_pred, ex["expected_intent"])
        results["haiku"]["correct"] += int(haiku_correct)
        results["haiku"]["total"] += 1

        row = {
            "message": ex["message"],
            "expected_intent": ex["expected_intent"],
            "haiku_intent": haiku_pred.get("intent"),
            "haiku_correct": haiku_correct,
        }

        if local_model is not None:
            local_pred, latency_ms = classify_with_local_model(local_model, local_tokenizer, ex["message"])
            local_correct = score(local_pred, ex["expected_intent"])
            results["local"]["correct"] += int(local_correct)
            results["local"]["total"] += 1
            results["local"]["latency_ms"].append(latency_ms)
            row["local_intent"] = local_pred.get("intent")
            row["local_correct"] = local_correct
            row["local_latency_ms"] = round(latency_ms, 1)

        per_example.append(row)

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "haiku_accuracy": results["haiku"]["correct"] / max(1, results["haiku"]["total"]),
        "local_accuracy": (
            results["local"]["correct"] / max(1, results["local"]["total"]) if local_model else None
        ),
        "local_avg_latency_ms": (
            sum(results["local"]["latency_ms"]) / max(1, len(results["local"]["latency_ms"]))
            if local_model
            else None
        ),
        "n_examples": len(test_set),
        "per_example": per_example,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- Summary ---")
    print(f"Examples evaluated: {report['n_examples']}")
    print(f"Haiku accuracy:  {report['haiku_accuracy']:.1%}")
    if report["local_accuracy"] is not None:
        print(f"Local accuracy:  {report['local_accuracy']:.1%}")
        print(f"Local avg latency: {report['local_avg_latency_ms']:.0f} ms  (vs. an API round-trip for Haiku)")
    print(f"Full report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
