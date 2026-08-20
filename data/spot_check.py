"""
Sample and pretty-print a handful of generated examples per language mode,
so you can eyeball quality before scaling up the full data generation run.

This is meant to be run right after a small `data/generate_dataset.py`
batch (e.g. --count 5 --per-intent-per-menu 2), before you commit to
spending the API budget on the full 3,000-5,000 example run.

Run:
    python data/spot_check.py --in data/raw/examples.jsonl --per-mode 5

Or narrow to one intent/mode while iterating on a specific prompt tweak:
    python data/spot_check.py --in data/raw/examples.jsonl --per-mode 8 \\
        --intent route_printer --mode mixed
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.schema import INTENT_NAMES  # noqa: E402
from data.generate_dataset import CREATIVE_SLOTS  # noqa: E402

try:
    from common.schema import LANGUAGE_MODES
except ImportError:  # older schema.py without the language-mode addition
    LANGUAGE_MODES = ("he", "en", "mixed")

# ANSI colors — purely cosmetic, degrade harmlessly if the terminal doesn't
# support them.
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"

MODE_COLOR = {"he": CYAN, "en": GREEN, "mixed": YELLOW}


def isolate_for_display(text: str) -> str:
    """Wrap text in Unicode bidi isolate marks so Hebrew renders correctly
    even when printed next to ASCII labels/punctuation in a terminal.
    Mirrors evaluate.py's helper of the same name — kept independent here
    so this script has no runtime dependency on the (heavier) eval module.
    """
    FSI, PDI = "\u2068", "\u2069"
    return f"{FSI}{text}{PDI}"


def load_examples(path: Path) -> list[dict]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            examples.append(json.loads(line))
    return examples


def group_by_mode(examples: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        mode = ex.get("language_mode", "unknown")
        grouped[mode].append(ex)
    return grouped


def print_example(ex: dict, index: int) -> None:
    mode = ex.get("language_mode", "unknown")
    color = MODE_COLOR.get(mode, DIM)
    label = ex["label"]

    print(f"{DIM}[{index}]{RESET} {color}{BOLD}{mode.upper()}{RESET}  {DIM}intent:{RESET} {label['intent']}")
    print(f"  message: {isolate_for_display(ex['message'])}")
    slots = label.get("slots", {})
    if slots:
        slot_str = ", ".join(f"{k}={isolate_for_display(str(v))}" for k, v in slots.items())
        print(f"  slots:   {slot_str}")
    else:
        print(f"  slots:   {DIM}(none){RESET}")
    print()


def sanity_flags(ex: dict) -> list[str]:
    """Cheap heuristics to flag examples worth a closer look — not a
    substitute for reading them, just a way to sort attention."""
    flags = []
    label = ex["label"]
    if label["intent"] not in INTENT_NAMES:
        flags.append(f"UNKNOWN INTENT: {label['intent']}")
    if label["intent"] != "unclear" and not label.get("slots"):
        flags.append("missing slots for a non-unclear intent")
    mode = ex.get("language_mode")
    message = ex.get("message", "")
    has_hebrew = any("\u0590" <= ch <= "\u05FF" for ch in message)
    has_latin = any(ch.isascii() and ch.isalpha() for ch in message)
    if mode == "he" and has_latin and not has_hebrew:
        flags.append("labeled 'he' but no Hebrew characters found")
    if mode == "en" and has_hebrew:
        flags.append("labeled 'en' but contains Hebrew characters")
    if mode == "mixed" and not (has_hebrew and has_latin):
        flags.append("labeled 'mixed' but doesn't contain both scripts")
    return flags


def normalize_value(text: str) -> str:
    """Loose normalization for near-duplicate detection: lowercase, collapse
    whitespace, strip trailing punctuation. No embeddings — just enough to
    catch the same value with different casing/trailing punctuation."""
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text.rstrip(".,!?;:'\"\u05be")


def opening_phrase(message: str, n_words: int) -> str:
    words = message.strip().split()
    return normalize_value(" ".join(words[:n_words]))


def check_value_repetition(examples: list[dict], threshold: int) -> bool:
    """Per creative-slot intent, report unique/total invented values and flag
    any (near-duplicate, normalized) value repeated beyond `threshold`.
    Returns True if anything was flagged."""
    print(f"{BOLD}Invented slot-value repetition (creative-slot intents){RESET}")
    any_flagged = False
    for intent, slot_name in CREATIVE_SLOTS.items():
        values = [
            ex["label"]["slots"][slot_name]
            for ex in examples
            if ex["label"]["intent"] == intent and slot_name in ex["label"].get("slots", {})
        ]
        if not values:
            print(f"  {intent} ({slot_name}): no examples with this slot present")
            continue
        counts = Counter(normalize_value(v) for v in values)
        repeats = [(v, c) for v, c in counts.most_common() if c > threshold]
        print(f"  {intent} ({slot_name}): {len(counts)} unique / {len(values)} total")
        for value, count in repeats:
            any_flagged = True
            print(f"    {RED}⚠ repeated {count}x:{RESET} {value!r}")
    print()
    return any_flagged


def check_phrasing_repetition(examples: list[dict], threshold: int, n_words: int) -> bool:
    """Per intent, report unique/total message opening phrases (first
    `n_words` words, normalized) and flag any repeated beyond `threshold`."""
    print(f"{BOLD}Message opening-phrase repetition (all intents){RESET}")
    any_flagged = False
    for intent in INTENT_NAMES:
        bucket = [ex for ex in examples if ex["label"]["intent"] == intent]
        if not bucket:
            continue
        counts = Counter(opening_phrase(ex["message"], n_words) for ex in bucket)
        repeats = [(p, c) for p, c in counts.most_common() if c > threshold]
        print(f"  {intent}: {len(counts)} unique openings / {len(bucket)} total")
        for phrase, count in repeats:
            any_flagged = True
            print(f"    {RED}⚠ opening repeated {count}x:{RESET} {phrase!r}...")
    print()
    return any_flagged


def print_language_mode_table(examples: list[dict]) -> None:
    """Print a per-intent breakdown of language_mode counts/percentages, to
    check the global he/en/mixed ratio holds up intent-by-intent too."""
    print(f"{BOLD}Per-intent language-mode breakdown{RESET}")
    header = f"  {'intent':22s}" + "".join(f"{m:>16s}" for m in LANGUAGE_MODES) + f"{'total':>8s}"
    print(header)
    for intent in INTENT_NAMES:
        bucket = [ex for ex in examples if ex["label"]["intent"] == intent]
        total = len(bucket)
        if not total:
            continue
        counts = Counter(ex.get("language_mode", "unknown") for ex in bucket)
        cells = "".join(
            f"{counts.get(m, 0):>6d} ({counts.get(m, 0) / total * 100:4.0f}%)" for m in LANGUAGE_MODES
        )
        print(f"  {intent:22s}{cells}{total:>8d}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=Path, default=Path("data/raw/examples.jsonl"))
    parser.add_argument("--per-mode", type=int, default=5, help="How many examples to show per language mode")
    parser.add_argument("--intent", default=None, help="Filter to a single intent, e.g. route_printer")
    parser.add_argument("--mode", default=None, choices=LANGUAGE_MODES, help="Filter to a single language mode")
    parser.add_argument("--seed", type=int, default=None, help="Omit for a fresh random sample each run")
    parser.add_argument(
        "--check-repetition",
        action="store_true",
        help="Instead of sampling examples, analyze invented-value/phrasing repetition and "
        "per-intent language-mode balance across the whole file.",
    )
    parser.add_argument(
        "--repetition-threshold",
        type=int,
        default=4,
        help="Flag a (normalized) value/opening phrase repeated more than this many times",
    )
    parser.add_argument(
        "--opening-words", type=int, default=4, help="Number of leading words compared for phrasing repetition"
    )
    args = parser.parse_args()

    if not args.in_path.exists():
        print(f"No such file: {args.in_path}. Run data/generate_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    examples = load_examples(args.in_path)
    if args.intent:
        examples = [ex for ex in examples if ex["label"]["intent"] == args.intent]
    if not examples:
        print("No examples match the given filters.", file=sys.stderr)
        sys.exit(1)

    if args.check_repetition:
        value_flagged = check_value_repetition(examples, args.repetition_threshold)
        phrasing_flagged = check_phrasing_repetition(examples, args.repetition_threshold, args.opening_words)
        print_language_mode_table(examples)
        if value_flagged or phrasing_flagged:
            print(f"{RED}Repetition flags found above — review before scaling up.{RESET}")
        else:
            print(f"{GREEN}No repetition beyond the threshold at this sample size.{RESET}")
        return

    rng = random.Random(args.seed)
    grouped = group_by_mode(examples)

    modes_to_show = [args.mode] if args.mode else list(LANGUAGE_MODES) + (
        ["unknown"] if "unknown" in grouped else []
    )

    total_flagged = 0
    index = 0
    for mode in modes_to_show:
        bucket = grouped.get(mode, [])
        if not bucket:
            continue
        sample = rng.sample(bucket, k=min(args.per_mode, len(bucket)))
        print(f"{BOLD}{'=' * 60}{RESET}")
        print(f"{BOLD}{mode.upper()}  ({len(bucket)} total, showing {len(sample)}){RESET}")
        print(f"{BOLD}{'=' * 60}{RESET}\n")
        for ex in sample:
            index += 1
            print_example(ex, index)
            flags = sanity_flags(ex)
            if flags:
                total_flagged += 1
                for flag in flags:
                    print(f"  {RED}⚠ {flag}{RESET}")
                print()

    print(f"{BOLD}{'-' * 60}{RESET}")
    print(f"Shown: {index} examples across {len(modes_to_show)} mode(s).")
    if total_flagged:
        print(f"{RED}{total_flagged} example(s) flagged by heuristics — worth a closer look.{RESET}")
    else:
        print(f"{GREEN}No heuristic flags on this sample.{RESET}")
    print(
        f"{DIM}Note: heuristic flags catch obvious mislabeling (wrong script for the "
        f"declared mode, missing slots), not phrasing quality or correctness — "
        f"still read the messages yourself.{RESET}"
    )


if __name__ == "__main__":
    main()