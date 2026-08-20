"""
Generate labeled training examples by prompting Claude to phrase requests
against the synthetic menus, in the style a real restaurant manager would.

This is a "teacher model" data generation step (related to what's sometimes
called Self-Instruct / distillation in the literature — worth citing that
by name in your project README, it's a real, established technique, not a
workaround). Claude (the teacher, large and capable) produces many diverse,
realistic examples; a much smaller model (the student) is later fine-tuned
on them to imitate this narrow skill cheaply and locally.

Requires: ANTHROPIC_API_KEY in the environment.

Run:
    python data/generate_dataset.py \\
        --menus data/raw/menus.jsonl \\
        --out data/raw/examples.jsonl \\
        --per-intent-per-menu 6

This will make real API calls — start with a small --per-intent-per-menu
value and a handful of menus to sanity check the output before scaling up
to your full 3,000-5,000 example target.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.schema import INTENTS, LANGUAGE_MODES, slots_for  # noqa: E402

try:
    import anthropic
except ImportError:
    print("Run: pip install anthropic", file=sys.stderr)
    raise

MODEL = "claude-sonnet-4-6"  # quality matters more than cost for teacher data

# Default language-mode split. SmartServe is built for the Israeli market —
# Hebrew is the primary language managers actually type in, not a secondary
# one, so it gets the largest share by default; the rest is a realistic
# code-switch rate plus a pure-English minority. Configurable via CLI flags
# below rather than hardcoded, but these are the sane defaults if you don't
# pass anything.
DEFAULT_HEBREW_FRACTION = 0.5
DEFAULT_CODESWITCH_FRACTION = 0.3
# (english fraction is whatever's left: 1 - hebrew - codeswitch, ~0.2 by default)

GENERATION_STYLE_RULES = """\
- Vary phrasing style across calls: sometimes terse, sometimes conversational,
  sometimes with typos or casual slang, sometimes formal.
- Follow the language mode instruction given to you exactly — don't default
  to English regardless of what's asked.
- Refer to the item the way a busy manager actually would — not always the
  exact catalog name (e.g. "the margherita" not "Margherita Pizza", or a
  shortened/misspelled version).
"""

GENERATION_SYSTEM_PROMPT = f"""\
You are generating training data for a restaurant-management AI assistant.
You will be given: a restaurant's menu, one target intent, the slot values
to express, and a required language mode. Produce ONE realistic,
natural-sounding message a real restaurant manager might type into a chat
box to request this action, following the language mode exactly.

Rules:
{GENERATION_STYLE_RULES}\
- Do NOT include the JSON answer in your message text. Output ONLY the
  manager's message, nothing else, no quotes around it, no explanation.
"""

# Intents where one slot value is better invented by the teacher model itself,
# as part of the SAME call that writes the message, instead of being decided
# here beforehand and handed to Claude as a fixed string to work around —
# that ordering produces artificial repetition (the message is forced to
# match a static label) and there's no reason to spend a second API call on
# it. sample_slot_values() leaves these as `None` placeholders; main() below
# fills them in from the same response that produces the message.
CREATIVE_SLOTS: dict[str, str] = {
    "update_description": "new_description",
    "create_item": "description",
    "update_item_name": "new_name",
}


def build_creative_system_prompt(slot_name: str) -> str:
    """Variant of GENERATION_SYSTEM_PROMPT for CREATIVE_SLOTS intents: asks
    Claude to invent the slot value and the message together in one JSON
    response, so the label and the message stay naturally consistent."""
    return f"""\
You are generating training data for a restaurant-management AI assistant.
You will be given: a restaurant's menu, one target intent, any already-fixed
slot values, and a required language mode. Produce ONE realistic,
natural-sounding message a real restaurant manager might type, following the
language mode exactly.

Rules:
{GENERATION_STYLE_RULES}\
- You must also invent the exact value of the "{slot_name}" slot yourself —
  a realistic, varied value a real manager would actually use, not a generic
  placeholder — and the message must reflect that exact value.
- Output ONLY a JSON object, no prose, no markdown fences, no text before or
  after it:
  {{"message": "<the manager's message>", "{slot_name}": "<the value you invented>"}}
"""


def build_json_output_reminder(slot_name: str) -> str:
    """A closing reminder appended AFTER the phrasing-style directive in the
    user turn, so the JSON-only contract is the last instruction the teacher
    reads before generating — not just stated once, earlier, in the system
    prompt. Added because update_description saw a small uptick in malformed
    JSON responses once a phrasing directive started being appended after
    the JSON rules in build_creative_system_prompt; reordering so JSON is the
    most recent instruction is the more robust fix regardless of which
    specific phrasing style was involved."""
    return (
        f'Reminder: your entire response must be ONLY the JSON object '
        f'{{"message": "...", "{slot_name}": "..."}} — no prose, no markdown '
        "fences, nothing before or after it, regardless of the phrasing style above."
    )

# Explicit per-mode instructions handed to the teacher model, so the language
# mix is a controlled parameter instead of "mix languages naturally" and
# whatever Claude feels like doing. "mixed" is weighted toward the dominant
# real-world pattern (Hebrew-matrix sentence, English item/price/technical
# term) but occasionally goes the other way too, since both happen.
LANGUAGE_MODE_DIRECTIVES: dict[str, str] = {
    "he": (
        "Language mode: HEBREW. Write the entire message in Hebrew (Hebrew "
        "script), the way an Israeli manager actually types day to day. Do "
        "not insert English words or Latin script."
    ),
    "en": (
        "Language mode: ENGLISH. Write the entire message in English, no "
        "Hebrew words or script at all."
    ),
    "mixed": (
        "Language mode: CODE-SWITCHED (Hebrew/English). Real Israeli "
        "managers constantly mix languages mid-sentence. Produce ONE of "
        "these realistic patterns:\n"
        "  - (most common) A Hebrew-matrix sentence with the item name, "
        "price/currency, or a technical term (e.g. printer, kitchen, stock) "
        'kept in English, e.g. "תעדכן את המחיר של ה-Margherita ל-55 שקל".\n'
        "  - (less common) An English-matrix sentence with a Hebrew word or "
        'prefix mixed in (e.g. ל-, ב-, את), e.g. "route bar tickets ל-kitchen '
        'printer".\n'
        "Keep the switch at the word/phrase level (item names, prices, "
        "technical terms) — don't alternate whole clauses back and forth."
    ),
}


def sample_language_mode(
    rng: random.Random, hebrew_fraction: float, codeswitch_fraction: float
) -> str:
    """Pick he/en/mixed for one example according to the configured ratio."""
    r = rng.random()
    if r < hebrew_fraction:
        return "he"
    if r < hebrew_fraction + codeswitch_fraction:
        return "mixed"
    return "en"


# Short, single-slot-swap intents ("update the price of X to Y", "rename X to
# Y") kept converging onto 2-3 canonical openings per language even with
# GENERATION_STYLE_RULES's generic "vary phrasing" instruction — there just
# aren't many structurally distinct ways to say something that short, unless
# the teacher is explicitly pushed toward a specific register per call.
# Multi-slot/contextual intents (route_printer, update_stock, delete_item)
# don't have this problem and are deliberately left alone.
PHRASING_STYLE_DIRECTIVES: list[str] = [
    "Phrase it as a direct imperative/command, e.g. 'Update the price of X to Y'.",
    "Phrase it as a casual aside/FYI, mentioning the change in passing rather than as a formal request.",
    "Phrase it as a question, e.g. 'Can you change...?' or 'Could we update...?'.",
    "Lead with the item/subject first, then the action, e.g. 'The margherita — can you bump the price up?'.",
    "Keep it very terse/shorthand, like a quick text message, minimal words.",
    "Attach a reason or context before making the request, e.g. 'Costs went up, so change the price of X to Y'.",
]

PHRASING_VARIED_INTENTS: set[str] = {"update_price", "update_item_name", "update_description"}


def sample_phrasing_style(rng: random.Random) -> str:
    """Pick a phrasing register for one example, for the intents prone to
    opening-phrase convergence (see PHRASING_VARIED_INTENTS)."""
    return rng.choice(PHRASING_STYLE_DIRECTIVES)


def build_user_prompt(
    menu: dict,
    intent_name: str,
    slot_values: dict,
    language_mode: str,
    phrasing_directive: str | None = None,
    output_reminder: str | None = None,
) -> str:
    item_lines = "\n".join(f"- {it['name']} (${it['price']:.0f})" for it in menu["items"])
    phrasing_block = f"Phrasing style for this message: {phrasing_directive}\n\n" if phrasing_directive else ""
    # output_reminder (when given) goes AFTER phrasing_block so it's the last
    # instruction the teacher reads before "Generate the manager's message
    # now." — see build_json_output_reminder()'s docstring for why order matters.
    reminder_block = f"{output_reminder}\n\n" if output_reminder else ""
    return (
        f"Restaurant: {menu['business_name']} ({menu['cuisine']})\n"
        f"Menu:\n{item_lines}\n\n"
        f"Target intent: {intent_name}\n"
        f"Slot values to express in the message: {json.dumps(slot_values, ensure_ascii=False)}\n\n"
        f"{LANGUAGE_MODE_DIRECTIVES[language_mode]}\n\n"
        f"{phrasing_block}"
        f"{reminder_block}"
        "Generate the manager's message now."
    )


def sample_slot_values(menu: dict, intent_name: str, rng: random.Random) -> dict:
    item = rng.choice(menu["items"])
    printer = rng.choice(menu["printers"])
    values: dict = {}
    if intent_name == "update_price":
        values = {"item_ref": item["name"], "new_price": round(item["price"] * rng.uniform(0.8, 1.3), 0)}
    elif intent_name == "create_item":
        values = {
            "item_name": f"{rng.choice(['Spicy', 'Grilled', 'House', 'Seasonal'])} {rng.choice(['Salad', 'Sandwich', 'Bowl', 'Platter'])}",
            "price": round(rng.uniform(25, 90), 0),
        }
        # "description" is optional per common/schema.py — sometimes omitted
        # entirely, and when present, its value is invented by the teacher
        # model itself (see CREATIVE_SLOTS), not a fixed string.
        if rng.random() < 0.7:
            values["description"] = None
    elif intent_name == "delete_item":
        values = {"item_ref": item["name"]}
    elif intent_name == "update_description":
        # new_description is invented by the teacher model (see CREATIVE_SLOTS).
        values = {"item_ref": item["name"], "new_description": None}
    elif intent_name == "update_item_name":
        # new_name is invented by the teacher model (see CREATIVE_SLOTS).
        values = {"item_ref": item["name"], "new_name": None}
    elif intent_name == "route_printer":
        target = rng.choice([item["name"], item["category"]])
        values = {"target_ref": target, "printer_ref": printer}
    elif intent_name == "update_stock":
        values = {"item_ref": item["name"], "stock_status": rng.choice(["out_of_stock", "in_stock"])}
    elif intent_name == "unclear":
        values = {}
    return values


# "unclear" examples are sampled from fixed pools + light variation rather
# than generated per-menu, since they're intentionally menu-agnostic and
# off-topic or ambiguous. Real ambiguous Hebrew phrasing (slang, informal,
# vague requests) needs its own pool — "unclear" has to work as well in
# Hebrew as the other intents do, not be an English-only afterthought.
UNCLEAR_POOL_EN = [
    "can you help me with something",
    "what's the deal with yesterday's orders",
    "make things better for the weekend rush",
    "I need to talk to someone about scheduling",
    "why does the system keep logging me out",
    "not sure what to do with the new supplier invoice",
]

UNCLEAR_POOL_HE = [
    "תבדוק לי מה קורה עם המלאי בכלל",
    "תעשה משהו עם התפריט, זה בלגן שלם",
    "אני לא יודע, תסדר את זה כבר",
    "מה קורה עם ההזמנות של אתמול בכלל",
    "צריך לדבר עם מישהו על הסידור עבודה של השבוע",
    "למה המערכת כל הזמן מנתקת אותי",
    "תטפל בזה כבר, נמאס לי",
    "יאללה תעשה סדר פה, זה בלאגן",
]

UNCLEAR_POOL_MIXED = [
    "תבדוק את ה-inventory כבר, זה בלגן",
    "can you תעשה סדר with the menu already",
    "משהו לא תקין עם ה-printer, תבדוק",
    "need to talk to someone על ה-scheduling",
]

UNCLEAR_POOLS: dict[str, list[str]] = {
    "he": UNCLEAR_POOL_HE,
    "en": UNCLEAR_POOL_EN,
    "mixed": UNCLEAR_POOL_MIXED,
}
assert set(UNCLEAR_POOLS) == set(LANGUAGE_MODES), "UNCLEAR_POOLS must cover every language mode"


def generate_unclear_message(rng: random.Random, language_mode: str) -> str:
    return rng.choice(UNCLEAR_POOLS[language_mode])


# The SDK's own built-in retries (anthropic.Anthropic(max_retries=...)) retry
# silently with no visibility into how often it's happening during a
# multi-hour run, so we disable those (max_retries=0 on the client, below)
# and do our own retry loop here instead, logging every retry. Exception
# types are the SDK's own classification (anthropic.RateLimitError etc.),
# not reinvented — this is just an outer harness for visibility + control
# over attempt count/backoff during the full-scale run.
RETRYABLE_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,  # 529 — transient, seen in practice during the full-scale run
)


def call_with_retry(fn, *, max_attempts: int = 5, base_delay: float = 2.0, **kwargs):
    """Call fn(**kwargs) (an Anthropic API call), retrying transient failures
    (rate limits, timeouts, connection errors, 5xx) with exponential backoff.
    Raises the last exception if all attempts are exhausted."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(**kwargs)
        except RETRYABLE_EXCEPTIONS as e:
            if attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"  WARN: transient API error ({type(e).__name__}: {e}), "
                f"retry {attempt}/{max_attempts - 1} in {delay:.0f}s...",
                file=sys.stderr,
            )
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--menus", type=Path, default=Path("data/raw/menus.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/examples.jsonl"))
    parser.add_argument("--per-intent-per-menu", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--intents",
        nargs="+",
        default=None,
        metavar="INTENT",
        help="Restrict generation to specific intents (space-separated), e.g. "
        "--intents update_price update_item_name. Default: all intents. Useful "
        "for cheaply re-verifying a prompt tweak against a subset.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.15,
        help="Fraction of menus (by business_id order) reserved as unseen — "
        "these are skipped here and used only in evaluation/evaluate.py.",
    )
    parser.add_argument(
        "--hebrew-fraction",
        type=float,
        default=DEFAULT_HEBREW_FRACTION,
        help="Fraction of examples generated as pure Hebrew (default: SmartServe's "
        "Israeli-market ratio, ~0.5 — Hebrew is the primary language here, not "
        "a secondary one).",
    )
    parser.add_argument(
        "--codeswitch-fraction",
        type=float,
        default=DEFAULT_CODESWITCH_FRACTION,
        help="Fraction of examples generated as code-switched Hebrew/English "
        "(default ~0.3). The remainder (1 - hebrew - codeswitch) is pure English.",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first.", file=sys.stderr)
        sys.exit(1)

    if args.hebrew_fraction < 0 or args.codeswitch_fraction < 0 or (
        args.hebrew_fraction + args.codeswitch_fraction
    ) > 1.0 + 1e-9:
        print(
            "--hebrew-fraction + --codeswitch-fraction must be between 0 and 1.0 "
            "(the remainder becomes the English fraction).",
            file=sys.stderr,
        )
        sys.exit(1)

    target_intents = list(args.intents) if args.intents else list(INTENTS)
    unknown = [name for name in target_intents if name not in INTENTS]
    if unknown:
        print(f"Unknown intent(s) in --intents: {unknown}. Valid: {list(INTENTS)}", file=sys.stderr)
        sys.exit(1)

    menus = [json.loads(line) for line in args.menus.read_text(encoding="utf-8").splitlines()]
    holdout_count = max(1, int(len(menus) * args.holdout_fraction))
    train_menus = menus[:-holdout_count]
    print(f"{len(train_menus)} menus for training data, {holdout_count} held out for evaluation")

    client = anthropic.Anthropic(max_retries=0)  # we retry ourselves, see call_with_retry
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: a (business_id, intent, gen_index) triple uniquely
    # identifies one generation slot in the loop below. If --out already has
    # some of these, skip regenerating them so a crash/interrupt partway
    # through only costs the time since the last write, not the whole run.
    completed: set[tuple[str, str, int]] = set()
    resuming = args.out.exists()
    if resuming:
        for line in args.out.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                completed.add((rec["business_id"], rec["label"]["intent"], rec["gen_index"]))
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a truncated last line from a hard kill
        print(f"Resuming from {args.out}: {len(completed)} example(s) already done, skipping those.")

    written = 0
    skipped = 0
    with args.out.open("a" if resuming else "w", encoding="utf-8") as f:
        for menu in train_menus:
            for intent_name in target_intents:
                for gen_index in range(args.per_intent_per_menu):
                    if (menu["business_id"], intent_name, gen_index) in completed:
                        continue
                    language_mode = sample_language_mode(
                        rng, args.hebrew_fraction, args.codeswitch_fraction
                    )
                    if intent_name == "unclear":
                        message = generate_unclear_message(rng, language_mode)
                        slot_values = {}
                    else:
                        slot_values = sample_slot_values(menu, intent_name, rng)
                        creative_slot = CREATIVE_SLOTS.get(intent_name)
                        needs_invention = (
                            creative_slot is not None
                            and creative_slot in slot_values
                            and slot_values[creative_slot] is None
                        )
                        phrasing_directive = (
                            sample_phrasing_style(rng) if intent_name in PHRASING_VARIED_INTENTS else None
                        )
                        # Don't show the teacher a placeholder null for the
                        # slot it's about to invent.
                        prompt_slot_values = {k: v for k, v in slot_values.items() if v is not None}
                        output_reminder = build_json_output_reminder(creative_slot) if needs_invention else None
                        user_prompt = build_user_prompt(
                            menu, intent_name, prompt_slot_values, language_mode, phrasing_directive, output_reminder
                        )

                        if needs_invention:
                            response = call_with_retry(
                                client.messages.create,
                                model=MODEL,
                                max_tokens=250,
                                system=build_creative_system_prompt(creative_slot),
                                messages=[{"role": "user", "content": user_prompt}],
                            )
                            raw_text = response.content[0].text.strip()
                            try:
                                parsed = json.loads(raw_text)
                                message = parsed["message"].strip()
                                slot_values[creative_slot] = parsed[creative_slot].strip()
                            except (json.JSONDecodeError, KeyError, AttributeError):
                                print(
                                    f"  WARN: malformed creative-slot response for "
                                    f"{intent_name} (phrasing_directive={phrasing_directive!r}), "
                                    f"skipping this example",
                                    file=sys.stderr,
                                )
                                skipped += 1
                                continue
                        else:
                            response = call_with_retry(
                                client.messages.create,
                                model=MODEL,
                                max_tokens=200,
                                system=GENERATION_SYSTEM_PROMPT,
                                messages=[{"role": "user", "content": user_prompt}],
                            )
                            message = response.content[0].text.strip()

                    example = {
                        "business_id": menu["business_id"],
                        "gen_index": gen_index,
                        "message": message,
                        "language_mode": language_mode,
                        "label": {
                            "intent": intent_name,
                            "confidence": 1.0,
                            "slots": slot_values,
                        },
                    }
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    f.flush()
                    written += 1
                    if written % 50 == 0:
                        print(f"  {written} examples written...")

    print(f"Done. Wrote {written} labeled examples to {args.out}")
    if skipped:
        print(f"Skipped {skipped} example(s) due to malformed creative-slot responses.")
    print("Next: python data/split_dataset.py to create train/val splits.")


if __name__ == "__main__":
    main()
