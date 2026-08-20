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
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.schema import INTENTS, LANGUAGE_MODES, OUTPUT_SCHEMA_DESCRIPTION  # noqa: E402
from data.generate_dataset import (  # noqa: E402
    CREATIVE_SLOTS,
    DEFAULT_CODESWITCH_FRACTION,
    DEFAULT_HEBREW_FRACTION,
    PHRASING_VARIED_INTENTS,
    build_creative_system_prompt,
    build_json_output_reminder,
    build_user_prompt,
    generate_unclear_message,
    sample_language_mode,
    sample_phrasing_style,
    sample_slot_values,
)
from data.spot_check import opening_phrase  # noqa: E402

import anthropic

# Unicode "first strong isolate" / "pop directional isolate" bidi controls.
# Hebrew message text printed inline with ASCII labels/punctuation (e.g.
# "[he] expected=update_price: <hebrew text>") can otherwise get visually
# reordered by the terminal/console's bidi algorithm in a way that glues
# trailing punctuation to the wrong end of the string. Isolating the RTL
# span keeps the surrounding ASCII layout stable and the Hebrew readable.
_ISOLATE_START = "\u2067"
_ISOLATE_END = "\u2069"


def isolate_for_display(text: str) -> str:
    """Wrap text that may be RTL (Hebrew) so printing it next to ASCII
    labels/numbers renders in a stable, readable order."""
    return f"{_ISOLATE_START}{text}{_ISOLATE_END}"


def build_test_set(
    menus: list[dict],
    n_per_intent: int,
    seed: int,
    hebrew_fraction: float = DEFAULT_HEBREW_FRACTION,
    codeswitch_fraction: float = DEFAULT_CODESWITCH_FRACTION,
) -> list[dict]:
    """Generate held-out test examples the same way training data was
    generated, but against menus the model never trained on."""
    import random

    rng = random.Random(seed)
    client = anthropic.Anthropic()
    test_set = []
    for menu in menus:
        for intent_name in INTENTS:
            for _ in range(n_per_intent):
                language_mode = sample_language_mode(rng, hebrew_fraction, codeswitch_fraction)
                slot_values = sample_slot_values(menu, intent_name, rng) if intent_name != "unclear" else {}
                if intent_name == "unclear":
                    message = generate_unclear_message(rng, language_mode)
                else:
                    creative_slot = CREATIVE_SLOTS.get(intent_name)
                    needs_invention = (
                        creative_slot is not None
                        and creative_slot in slot_values
                        and slot_values[creative_slot] is None
                    )
                    phrasing_directive = (
                        sample_phrasing_style(rng) if intent_name in PHRASING_VARIED_INTENTS else None
                    )
                    prompt_slot_values = {k: v for k, v in slot_values.items() if v is not None}
                    output_reminder = build_json_output_reminder(creative_slot) if needs_invention else None
                    prompt = build_user_prompt(
                        menu, intent_name, prompt_slot_values, language_mode, phrasing_directive, output_reminder
                    )
                    if needs_invention:
                        resp = client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=250,
                            system=build_creative_system_prompt(creative_slot),
                            messages=[{"role": "user", "content": prompt}],
                        )
                        raw_text = resp.content[0].text.strip()
                        try:
                            parsed = json.loads(raw_text)
                            message = parsed["message"].strip()
                            slot_values[creative_slot] = parsed[creative_slot].strip()
                        except (json.JSONDecodeError, KeyError, AttributeError):
                            # Keep the test example rather than dropping it —
                            # fall back to the raw (possibly non-JSON) text as
                            # the message; the missing slot value will simply
                            # not match, same as any other model miss.
                            message = raw_text
                    else:
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
                        "language_mode": language_mode,
                        "expected_intent": intent_name,
                        "expected_slots": slot_values,
                    }
                )
    return test_set



def _parse_json_response(text: str) -> dict | None:
    """Parse a model's JSON response, tolerating ```json ... ``` markdown
    fences around it. Haiku wraps its output in fences despite being told
    'no markdown fences' in the prompt; without stripping them, json.loads
    fails on every single call and silently falls back to 'unclear',
    which previously produced a bogus ~12% Haiku accuracy."""
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


def classify_with_haiku(client: anthropic.Anthropic, message: str) -> dict:
    system = "Classify this restaurant manager message. " + OUTPUT_SCHEMA_DESCRIPTION
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=system,
        messages=[{"role": "user", "content": message}],
    )
    parsed = _parse_json_response(resp.content[0].text)
    return parsed if parsed is not None else {"intent": "unclear", "confidence": 0.0, "slots": {}}


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
    parsed = _parse_json_response(text)
    if parsed is not None:
        return parsed, latency_ms
    return {"intent": "unclear", "confidence": 0.0, "slots": {}}, latency_ms


def score(predicted: dict, expected_intent: str) -> bool:
    return predicted.get("intent") == expected_intent


def _new_language_buckets() -> dict[str, dict]:
    return {lang: {"correct": 0, "total": 0} for lang in LANGUAGE_MODES}


def _accuracy_by_language(buckets: dict[str, dict]) -> dict[str, dict]:
    return {
        lang: {
            "accuracy": (b["correct"] / b["total"]) if b["total"] else None,
            "n": b["total"],
        }
        for lang, b in buckets.items()
    }


PHRASING_STRATA = ("template_match", "long_tail")


def load_dominant_openings(
    train_path: Path, opening_words: int, dominant_fraction: float
) -> dict[str, set[str]]:
    """Per intent, find the (normalized) message openings that make up at
    least `dominant_fraction` of that intent's examples in the training set
    — i.e. the template patterns data/spot_check.py's --check-repetition
    analysis surfaces. Reuses spot_check.py's own opening_phrase() so this
    stays consistent with however that script defines an "opening" instead
    of reinventing the normalization. Returns {} (nothing dominant) if
    train_path doesn't exist, rather than failing — this stratification is a
    bonus lens on the results, not a hard requirement to run evaluate.py."""
    if not train_path.exists():
        return {}
    examples = [
        json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    by_intent: dict[str, list[str]] = defaultdict(list)
    for ex in examples:
        by_intent[ex["label"]["intent"]].append(ex["message"])

    dominant: dict[str, set[str]] = {}
    for intent, messages in by_intent.items():
        counts = Counter(opening_phrase(m, opening_words) for m in messages)
        total = len(messages)
        dominant[intent] = {phrase for phrase, count in counts.items() if count / total >= dominant_fraction}
    return dominant


def classify_phrasing_stratum(
    message: str, intent: str, dominant_openings: dict[str, set[str]], opening_words: int
) -> str:
    """template_match if this message's opening matches one of the intent's
    dominant (over-represented) training openings, long_tail otherwise."""
    opening = opening_phrase(message, opening_words)
    if opening in dominant_openings.get(intent, set()):
        return "template_match"
    return "long_tail"


def _new_phrasing_buckets() -> dict[str, dict]:
    return {
        intent: {stratum: {"correct": 0, "total": 0} for stratum in PHRASING_STRATA} for intent in INTENTS
    }


def _accuracy_by_phrasing(buckets: dict[str, dict]) -> dict[str, dict]:
    return {
        intent: {
            stratum: {
                "accuracy": (b["correct"] / b["total"]) if b["total"] else None,
                "n": b["total"],
            }
            for stratum, b in strata.items()
        }
        for intent, strata in buckets.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--menus", type=Path, default=Path("data/raw/menus.jsonl"))
    parser.add_argument("--model-path", type=Path, default=Path("training/checkpoints/final"))
    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    parser.add_argument("--n-per-intent", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)  # different from data-gen seed
    parser.add_argument("--skip-local", action="store_true", help="Score Haiku only (no GPU needed)")
    parser.add_argument(
        "--hebrew-fraction",
        type=float,
        default=DEFAULT_HEBREW_FRACTION,
        help="Fraction of the held-out test set generated as pure Hebrew (default matches data/generate_dataset.py).",
    )
    parser.add_argument(
        "--codeswitch-fraction",
        type=float,
        default=DEFAULT_CODESWITCH_FRACTION,
        help="Fraction of the held-out test set generated as code-switched Hebrew/English.",
    )
    parser.add_argument(
        "--train-examples",
        type=Path,
        default=Path("data/raw/examples.jsonl"),
        help="Training examples file used to find each intent's 'dominant' (templated) "
        "message openings, per data/spot_check.py's --check-repetition analysis. Held-out "
        "test examples are scored separately as template_match vs long_tail against this. "
        "If the file doesn't exist, phrasing-stratum reporting is skipped (everything is "
        "reported as long_tail).",
    )
    parser.add_argument(
        "--opening-words",
        type=int,
        default=4,
        help="Number of leading words compared for the template_match/long_tail phrasing "
        "split (matches data/spot_check.py's --opening-words default).",
    )
    parser.add_argument(
        "--dominant-opening-fraction",
        type=float,
        default=0.05,
        help="An opening phrase used by at least this fraction of an intent's training "
        "examples counts as 'dominant'/templated for the phrasing-stratum breakdown.",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first.")
        sys.exit(1)

    dominant_openings = load_dominant_openings(
        args.train_examples, args.opening_words, args.dominant_opening_fraction
    )
    if dominant_openings:
        n_dominant = sum(len(v) for v in dominant_openings.values())
        print(
            f"Loaded {n_dominant} dominant opening pattern(s) across {len(dominant_openings)} "
            f"intent(s) from {args.train_examples} for template_match/long_tail scoring"
        )
    else:
        print(
            f"No dominant-opening data from {args.train_examples} (missing file, or nothing met "
            f"the {args.dominant_opening_fraction:.0%} threshold) — phrasing stratum will report "
            "everything as long_tail."
        )

    all_menus = [json.loads(line) for line in args.menus.read_text(encoding="utf-8").splitlines()]
    holdout_count = max(1, int(len(all_menus) * args.holdout_fraction))
    holdout_menus = all_menus[-holdout_count:]
    print(f"Evaluating against {len(holdout_menus)} menus never seen during training")

    test_set = build_test_set(
        holdout_menus,
        args.n_per_intent,
        args.seed,
        hebrew_fraction=args.hebrew_fraction,
        codeswitch_fraction=args.codeswitch_fraction,
    )
    print(f"Built {len(test_set)} held-out test examples")

    haiku_client = anthropic.Anthropic()

    results = {
        "haiku": {
            "correct": 0,
            "total": 0,
            "by_language": _new_language_buckets(),
            "by_phrasing": _new_phrasing_buckets(),
        },
        "local": {
            "correct": 0,
            "total": 0,
            "latency_ms": [],
            "by_language": _new_language_buckets(),
            "by_phrasing": _new_phrasing_buckets(),
        },
    }

    local_model, local_tokenizer = None, None
    if not args.skip_local:
        from unsloth import FastLanguageModel

        from training.train import _patch_triton_for_mingw_windows  # same Windows/gcc/triton fixes as training

        _patch_triton_for_mingw_windows()
        local_model, local_tokenizer = FastLanguageModel.from_pretrained(str(args.model_path))
        FastLanguageModel.for_inference(local_model)

    per_example = []
    for ex in test_set:
        lang = ex["language_mode"]
        intent = ex["expected_intent"]
        stratum = classify_phrasing_stratum(ex["message"], intent, dominant_openings, args.opening_words)
        haiku_pred = classify_with_haiku(haiku_client, ex["message"])
        haiku_correct = score(haiku_pred, intent)
        results["haiku"]["correct"] += int(haiku_correct)
        results["haiku"]["total"] += 1
        results["haiku"]["by_language"][lang]["correct"] += int(haiku_correct)
        results["haiku"]["by_language"][lang]["total"] += 1
        results["haiku"]["by_phrasing"][intent][stratum]["correct"] += int(haiku_correct)
        results["haiku"]["by_phrasing"][intent][stratum]["total"] += 1

        row = {
            "message": ex["message"],
            "language_mode": lang,
            "expected_intent": intent,
            "phrasing_stratum": stratum,
            "haiku_intent": haiku_pred.get("intent"),
            "haiku_correct": haiku_correct,
        }

        if local_model is not None:
            local_pred, latency_ms = classify_with_local_model(local_model, local_tokenizer, ex["message"])
            local_correct = score(local_pred, intent)
            results["local"]["correct"] += int(local_correct)
            results["local"]["total"] += 1
            results["local"]["latency_ms"].append(latency_ms)
            results["local"]["by_language"][lang]["correct"] += int(local_correct)
            results["local"]["by_language"][lang]["total"] += 1
            results["local"]["by_phrasing"][intent][stratum]["correct"] += int(local_correct)
            results["local"]["by_phrasing"][intent][stratum]["total"] += 1
            row["local_intent"] = local_pred.get("intent")
            row["local_correct"] = local_correct
            row["local_latency_ms"] = round(latency_ms, 1)

        per_example.append(row)

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "haiku_accuracy": results["haiku"]["correct"] / max(1, results["haiku"]["total"]),
        "haiku_accuracy_by_language": _accuracy_by_language(results["haiku"]["by_language"]),
        "haiku_accuracy_by_phrasing": _accuracy_by_phrasing(results["haiku"]["by_phrasing"]),
        "local_accuracy": (
            results["local"]["correct"] / max(1, results["local"]["total"]) if local_model else None
        ),
        "local_accuracy_by_language": (
            _accuracy_by_language(results["local"]["by_language"]) if local_model else None
        ),
        "local_accuracy_by_phrasing": (
            _accuracy_by_phrasing(results["local"]["by_phrasing"]) if local_model else None
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

    # Per-language breakdown. An aggregate accuracy figure can hide a model
    # that's great in English and much worse in Hebrew — a real problem for
    # an Israeli-market product — so this is reported every run, not just
    # buried in report.json. Table cells are language codes + percentages
    # (plain ASCII), so this is RTL-safe regardless of terminal bidi support.
    print("\n--- Accuracy by language_mode ---")
    print(f"  {'mode':<6} {'haiku':>8} {'local':>8} {'n':>5}")
    worst_local_gap = 0.0
    for lang in LANGUAGE_MODES:
        h = report["haiku_accuracy_by_language"][lang]
        loc = report["local_accuracy_by_language"][lang] if report["local_accuracy_by_language"] else None
        h_acc = f"{h['accuracy']:.1%}" if h["accuracy"] is not None else "n/a"
        l_acc = f"{loc['accuracy']:.1%}" if (loc and loc["accuracy"] is not None) else "n/a"
        print(f"  {lang:<6} {h_acc:>8} {l_acc:>8} {h['n']:>5}")
        if loc and loc["accuracy"] is not None and report["local_accuracy"] is not None:
            worst_local_gap = max(worst_local_gap, report["local_accuracy"] - loc["accuracy"])
    if worst_local_gap > 0.1:
        print(
            f"  WARNING: local model is >{worst_local_gap:.0%} worse on at least one "
            "language than its overall accuracy — check the breakdown above before "
            "trusting the aggregate number."
        )

    # Per-intent template_match vs long_tail breakdown. This is the direct
    # answer to "does the residual opening-phrase clustering (see
    # data/spot_check.py --check-repetition) actually hurt generalization, or
    # is it harmless?" — if accuracy on template_match and long_tail examples
    # is similar, the clustering is benign; if it diverges, the model may be
    # overfitting to the dominant phrasing rather than the underlying intent.
    print("\n--- Accuracy by phrasing stratum (template_match vs long_tail) ---")
    print(f"  {'intent':<22} {'haiku_tmpl':>10} {'haiku_tail':>10} {'local_tmpl':>10} {'local_tail':>10} {'n_tmpl':>7} {'n_tail':>7}")
    worst_phrasing_gap = 0.0
    worst_phrasing_intent = None
    for intent in INTENTS:
        h_tmpl = report["haiku_accuracy_by_phrasing"][intent]["template_match"]
        h_tail = report["haiku_accuracy_by_phrasing"][intent]["long_tail"]
        loc_by_phrasing = report["local_accuracy_by_phrasing"]
        l_tmpl = loc_by_phrasing[intent]["template_match"] if loc_by_phrasing else None
        l_tail = loc_by_phrasing[intent]["long_tail"] if loc_by_phrasing else None
        if h_tmpl["n"] == 0 and h_tail["n"] == 0:
            continue
        ht_acc = f"{h_tmpl['accuracy']:.1%}" if h_tmpl["accuracy"] is not None else "n/a"
        hl_acc = f"{h_tail['accuracy']:.1%}" if h_tail["accuracy"] is not None else "n/a"
        lt_acc = f"{l_tmpl['accuracy']:.1%}" if (l_tmpl and l_tmpl["accuracy"] is not None) else "n/a"
        ll_acc = f"{l_tail['accuracy']:.1%}" if (l_tail and l_tail["accuracy"] is not None) else "n/a"
        print(
            f"  {intent:<22} {ht_acc:>10} {hl_acc:>10} {lt_acc:>10} {ll_acc:>10} "
            f"{h_tmpl['n']:>7} {h_tail['n']:>7}"
        )
        if h_tmpl["accuracy"] is not None and h_tail["accuracy"] is not None:
            gap = abs(h_tmpl["accuracy"] - h_tail["accuracy"])
            if gap > worst_phrasing_gap:
                worst_phrasing_gap, worst_phrasing_intent = gap, intent
    if worst_phrasing_intent and worst_phrasing_gap > 0.15:
        print(
            f"  WARNING: '{worst_phrasing_intent}' shows a {worst_phrasing_gap:.0%} accuracy gap "
            "between template_match and long_tail phrasing — the residual opening-phrase "
            "clustering may be a real generalization gap here, not just cosmetic repetition."
        )
    elif worst_phrasing_intent:
        print(
            f"  No large template_match/long_tail gap detected (worst: {worst_phrasing_gap:.0%} on "
            f"'{worst_phrasing_intent}') — the residual phrasing clustering looks benign so far."
        )

    # A couple of misclassified examples per language, for a quick sanity
    # check while iterating. Hebrew message text is isolated so it doesn't
    # visually merge with the surrounding ASCII labels/punctuation.
    print("\n--- Sample misclassified examples (haiku, by language_mode) ---")
    for lang in LANGUAGE_MODES:
        misses = [r for r in per_example if r["language_mode"] == lang and not r["haiku_correct"]]
        if not misses:
            continue
        sample = misses[0]
        print(
            f"  [{lang}] expected={sample['expected_intent']} got={sample['haiku_intent']}: "
            f"{isolate_for_display(sample['message'])}"
        )

    print(f"\nFull report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
