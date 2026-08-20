"""
Re-run accuracy + latency evaluation for a quantized GGUF model against the
EXACT same held-out test examples used in the unquantized run (loaded from
an existing evaluation/results/report.json's "per_example" list), so the
comparison is apples-to-apples: no regenerated test set (no fresh Anthropic
API calls) and no Haiku re-scoring (Haiku's numbers are already final).

Run:
    python evaluation/evaluate_quantized.py \\
        --gguf inference_gguf/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf \\
        --out evaluation/results/report_quantized_q4_k_m.json \\
        --quant-label q4_k_m
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.schema import LANGUAGE_MODES, OUTPUT_SCHEMA_DESCRIPTION  # noqa: E402

INSTRUCTION = (
    "You classify a restaurant manager's message into a structured action for a POS system. "
    + OUTPUT_SCHEMA_DESCRIPTION
)

# Same shape as training/train.py's PROMPT_TEMPLATE, inlined here so this
# script has no dependency on unsloth/torch (llama.cpp only).
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def _parse_json_response(text: str) -> dict | None:
    """Same fence-tolerant parser as evaluation/evaluate.py's helper."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def classify(llm, message: str) -> tuple[dict, float]:
    prompt = PROMPT_TEMPLATE.format(instruction=INSTRUCTION, input=message, output="")
    start = time.time()
    output = llm(prompt, max_tokens=200, stop=["###"], temperature=0.0)
    latency_ms = (time.time() - start) * 1000
    parsed = _parse_json_response(output["choices"][0]["text"])
    if parsed is not None:
        return parsed, latency_ms
    return {"intent": "unclear", "confidence": 0.0, "slots": {}}, latency_ms


def _new_language_buckets() -> dict[str, dict]:
    return {lang: {"correct": 0, "total": 0} for lang in LANGUAGE_MODES}


def _accuracy_by_language(buckets: dict[str, dict]) -> dict[str, dict]:
    return {
        lang: {"accuracy": (b["correct"] / b["total"]) if b["total"] else None, "n": b["total"]}
        for lang, b in buckets.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation/results/report.json"),
        help="Existing unquantized-run report to reuse per_example messages/expected_intents from.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quant-label", default="q4_k_m")
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument(
        "--n-threads",
        type=int,
        default=4,
        help="llama.cpp is memory-bandwidth bound for single-sequence decoding on this machine: "
        "measured n_threads=4 fastest (~2.7s/call), n_threads=28 (all logical cores) ~2x slower "
        "(~5.9s/call) due to thread contention, not faster. Don't default to os.cpu_count().",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU. -1 offloads all layers (full GPU acceleration). "
        "0 (default) keeps the original CPU-only behavior.",
    )
    parser.add_argument(
        "--verbose-load",
        action="store_true",
        help="Print llama.cpp's own model-load logs (confirms how many layers were actually "
        "offloaded to GPU) instead of suppressing them.",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    examples = report["per_example"]
    print(f"Reusing {len(examples)} held-out examples from {args.report} (no regeneration, no Haiku re-call)")

    from llama_cpp import Llama

    print(f"Loading {args.gguf} (n_gpu_layers={args.n_gpu_layers})...")
    llm = Llama(
        model_path=str(args.gguf),
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        verbose=args.verbose_load,
    )

    correct = 0
    latencies = []
    by_language = _new_language_buckets()
    per_example_out = []
    for i, ex in enumerate(examples):
        pred, latency_ms = classify(llm, ex["message"])
        is_correct = pred.get("intent") == ex["expected_intent"]
        correct += int(is_correct)
        latencies.append(latency_ms)
        lang = ex["language_mode"]
        by_language[lang]["correct"] += int(is_correct)
        by_language[lang]["total"] += 1
        per_example_out.append(
            {
                "message": ex["message"],
                "language_mode": lang,
                "expected_intent": ex["expected_intent"],
                "phrasing_stratum": ex.get("phrasing_stratum"),
                "haiku_intent": ex.get("haiku_intent"),
                "haiku_correct": ex.get("haiku_correct"),
                "quantized_intent": pred.get("intent"),
                "quantized_correct": is_correct,
                "quantized_latency_ms": latency_ms,
            }
        )
        if (i + 1) % 20 == 0:
            running_avg = sum(latencies) / len(latencies)
            print(f"  {i + 1}/{len(examples)}... (running avg {running_avg:.0f}ms/example)")

    accuracy = correct / len(examples)
    avg_latency = sum(latencies) / len(latencies)
    by_lang_result = _accuracy_by_language(by_language)

    result = {
        "quant_label": args.quant_label,
        "gguf_path": str(args.gguf),
        "n_examples": len(examples),
        "quantized_accuracy": accuracy,
        "quantized_accuracy_by_language": by_lang_result,
        "quantized_avg_latency_ms": avg_latency,
        "per_example": per_example_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== Quantized ({args.quant_label}) results ===")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{len(examples)})")
    print(f"Avg latency: {avg_latency:.1f} ms/example")
    for lang, stats in by_lang_result.items():
        if stats["n"]:
            print(f"  {lang}: {stats['accuracy']:.1%} (n={stats['n']})")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
