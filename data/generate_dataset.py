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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.schema import INTENTS, slots_for  # noqa: E402

try:
    import anthropic
except ImportError:
    print("Run: pip install anthropic", file=sys.stderr)
    raise

MODEL = "claude-sonnet-4-6"  # quality matters more than cost for teacher data

GENERATION_SYSTEM_PROMPT = """\
You are generating training data for a restaurant-management AI assistant.
You will be given: a restaurant's menu, one target intent, and the slot
values to express. Produce ONE realistic, natural-sounding message a real
restaurant manager might type into a chat box to request this action.

Rules:
- Vary phrasing style across calls: sometimes terse, sometimes conversational,
  sometimes with typos or casual slang, sometimes formal.
- Mix languages naturally: sometimes English, sometimes Hebrew, sometimes
  Hebrew with the item name kept in English or vice versa (this is common in
  Israeli businesses).
- Refer to the item the way a busy manager actually would — not always the
  exact catalog name (e.g. "the margherita" not "Margherita Pizza", or a
  shortened/misspelled version).
- Do NOT include the JSON answer in your message text. Output ONLY the
  manager's message, nothing else, no quotes around it, no explanation.
"""


def build_user_prompt(menu: dict, intent_name: str, slot_values: dict) -> str:
    item_lines = "\n".join(f"- {it['name']} (${it['price']:.0f})" for it in menu["items"])
    return (
        f"Restaurant: {menu['business_name']} ({menu['cuisine']})\n"
        f"Menu:\n{item_lines}\n\n"
        f"Target intent: {intent_name}\n"
        f"Slot values to express in the message: {json.dumps(slot_values, ensure_ascii=False)}\n\n"
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
            "description": "a new addition to the menu",
        }
    elif intent_name == "delete_item":
        values = {"item_ref": item["name"]}
    elif intent_name == "update_description":
        values = {"item_ref": item["name"], "new_description": "updated description text"}
    elif intent_name == "update_item_name":
        values = {"item_ref": item["name"], "new_name": item["name"] + " Deluxe"}
    elif intent_name == "route_printer":
        target = rng.choice([item["name"], item["category"]])
        values = {"target_ref": target, "printer_ref": printer}
    elif intent_name == "update_stock":
        values = {"item_ref": item["name"], "stock_status": rng.choice(["out_of_stock", "in_stock"])}
    elif intent_name == "unclear":
        values = {}
    return values


def generate_unclear_message(rng: random.Random) -> str:
    # "unclear" examples are sampled from a fixed pool + light variation
    # rather than generated per-menu, since they're intentionally
    # menu-agnostic and off-topic or ambiguous.
    pool = [
        "can you help me with something",
        "what's the deal with yesterday's orders",
        "make things better for the weekend rush",
        "תבדוק לי מה קורה עם המלאי בכלל",
        "I need to talk to someone about scheduling",
        "why does the system keep logging me out",
    ]
    return rng.choice(pool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--menus", type=Path, default=Path("data/raw/menus.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/examples.jsonl"))
    parser.add_argument("--per-intent-per-menu", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.15,
        help="Fraction of menus (by business_id order) reserved as unseen — "
        "these are skipped here and used only in evaluation/evaluate.py.",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first.", file=sys.stderr)
        sys.exit(1)

    menus = [json.loads(line) for line in args.menus.read_text(encoding="utf-8").splitlines()]
    holdout_count = max(1, int(len(menus) * args.holdout_fraction))
    train_menus = menus[:-holdout_count]
    print(f"{len(train_menus)} menus for training data, {holdout_count} held out for evaluation")

    client = anthropic.Anthropic()
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for menu in train_menus:
            for intent_name in INTENTS:
                for _ in range(args.per_intent_per_menu):
                    if intent_name == "unclear":
                        message = generate_unclear_message(rng)
                        slot_values = {}
                    else:
                        slot_values = sample_slot_values(menu, intent_name, rng)
                        user_prompt = build_user_prompt(menu, intent_name, slot_values)
                        response = client.messages.create(
                            model=MODEL,
                            max_tokens=200,
                            system=GENERATION_SYSTEM_PROMPT,
                            messages=[{"role": "user", "content": user_prompt}],
                        )
                        message = response.content[0].text.strip()

                    example = {
                        "business_id": menu["business_id"],
                        "message": message,
                        "label": {
                            "intent": intent_name,
                            "confidence": 1.0,
                            "slots": slot_values,
                        },
                    }
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    written += 1
                    if written % 50 == 0:
                        print(f"  {written} examples written...")

    print(f"Done. Wrote {written} labeled examples to {args.out}")
    print("Next: python data/split_dataset.py to create train/val splits.")


if __name__ == "__main__":
    main()
