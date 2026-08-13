"""
Generate diverse synthetic restaurant menus.

Why this exists: if every training example references the same 10 menu
items, the model learns to recognize those 10 items, not the *pattern* of
how managers refer to items. Generating many different fake restaurants
with different cuisines, naming conventions, and languages is what forces
the model to generalize instead of memorize.

Run:
    python data/generate_menus.py --count 40 --out data/raw/menus.jsonl

Output: one JSON object per line, each a full synthetic restaurant menu.
Keep a handful of menus completely unused until evaluation/evaluate.py —
that's your generalization test, not just a train/test split of the same
distribution.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CUISINES = [
    ("Italian", ["Margherita Pizza", "Fettuccine Alfredo", "Tiramisu", "Bruschetta", "Osso Buco"]),
    ("Israeli", ["Shakshuka", "Sabich", "Hummus with Pita", "Malabi", "Jachnun"]),
    ("Japanese", ["Salmon Nigiri", "Chicken Katsu", "Miso Soup", "Gyoza", "Matcha Ice Cream"]),
    ("Mexican", ["Chicken Tacos", "Guacamole", "Quesadilla", "Churros", "Elote"]),
    ("American Diner", ["Cheeseburger", "Buffalo Wings", "Milkshake", "Mac and Cheese", "Apple Pie"]),
    ("Indian", ["Chicken Tikka Masala", "Garlic Naan", "Samosa", "Mango Lassi", "Palak Paneer"]),
    ("Cocktail Bar", ["Old Fashioned", "Negroni", "Mojito", "Espresso Martini", "Bar Nuts"]),
]

HEBREW_NAMES = {
    "Margherita Pizza": "פיצה מרגריטה",
    "Shakshuka": "שקשוקה",
    "Sabich": "סביח",
    "Hummus with Pita": "חומוס עם פיתה",
    "Cheeseburger": "צ'יזבורגר",
    "Chicken Tacos": "טאקו עוף",
    "Old Fashioned": "אולד פאשן",
}

CATEGORIES = ["Starters", "Mains", "Desserts", "Drinks", "Bar"]
PRINTERS = ["Kitchen Printer", "Bar Printer", "Dessert Station", "Expo Printer"]


def make_menu(business_id: int, rng: random.Random) -> dict:
    cuisine, base_items = rng.choice(CUISINES)
    business_name = f"{cuisine} Spot #{business_id}"
    items = []
    for item_name in base_items:
        price = round(rng.uniform(18, 120), 0)
        items.append(
            {
                "item_id": f"item_{business_id}_{len(items)}",
                "name": item_name,
                "name_he": HEBREW_NAMES.get(item_name),  # may be None — that's fine
                "price": price,
                "category": rng.choice(CATEGORIES),
                "printer": rng.choice(PRINTERS),
            }
        )
    return {
        "business_id": f"biz_{business_id}",
        "business_name": business_name,
        "cuisine": cuisine,
        "location_ids": [f"loc_{business_id}_1"],
        "printers": PRINTERS,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("data/raw/menus.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.count):
            menu = make_menu(i, rng)
            f.write(json.dumps(menu, ensure_ascii=False) + "\n")

    print(f"Wrote {args.count} synthetic menus to {args.out}")
    print(
        "Tip: hold out the last ~15% of business_ids (by index) for "
        "evaluation — don't let the training set see them."
    )


if __name__ == "__main__":
    main()
