"""
Shared intent schema for the SmartServe local agent model.

This is the single source of truth for what the model is being taught to
produce. It's imported by the data generator, the training script, and the
evaluator, so the three stay in sync by construction instead of by discipline.

Keep this list matched to whatever intents your SmartServe `services/ai`
Haiku classifier already supports, so the local model is a drop-in
first-tier replacement, not a new contract the rest of the system has to
learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IntentSpec:
    name: str
    description: str
    # Required slot names for this intent. The generator will always fill
    # these; the model is trained to always emit them (empty string if truly
    # not present in the utterance, but the generator shouldn't produce that
    # case for these intents since they're required for execution).
    required_slots: tuple[str, ...]
    # Optional slot names — may or may not appear depending on phrasing.
    optional_slots: tuple[str, ...] = field(default_factory=tuple)


INTENTS: dict[str, IntentSpec] = {
    "update_price": IntentSpec(
        name="update_price",
        description="Change the price of an existing menu item.",
        required_slots=("item_ref", "new_price"),
        optional_slots=("price_delta", "currency"),
    ),
    "create_item": IntentSpec(
        name="create_item",
        description="Add a new item to the menu.",
        required_slots=("item_name", "price"),
        optional_slots=("description", "category", "printer_ref"),
    ),
    "delete_item": IntentSpec(
        name="delete_item",
        description="Remove an existing item from the menu.",
        required_slots=("item_ref",),
    ),
    "update_description": IntentSpec(
        name="update_description",
        description="Change the description text of an existing item.",
        required_slots=("item_ref", "new_description"),
    ),
    "update_item_name": IntentSpec(
        name="update_item_name",
        description="Rename an existing item.",
        required_slots=("item_ref", "new_name"),
    ),
    "route_printer": IntentSpec(
        name="route_printer",
        description="Route an item or category to a specific printer.",
        required_slots=("target_ref", "printer_ref"),
    ),
    "update_stock": IntentSpec(
        name="update_stock",
        description="Mark an item as in stock / out of stock, or set a quantity.",
        required_slots=("item_ref", "stock_status"),
        optional_slots=("quantity",),
    ),
    "unclear": IntentSpec(
        name="unclear",
        description=(
            "The request doesn't map cleanly to any of the above — either "
            "genuinely ambiguous, multi-step, or outside this model's scope. "
            "The model should prefer this over guessing, since 'unclear' "
            "triggers a safe escalation to Haiku/Sonnet rather than a wrong "
            "silent action."
        ),
        required_slots=(),
    ),
}

INTENT_NAMES: tuple[str, ...] = tuple(INTENTS.keys())

# Language mode tag attached to every generated example (see data/generate_dataset.py)
# so accuracy can be sliced per-language at evaluation time. SmartServe's primary
# market is Israel, where Hebrew is the default day-to-day language and managers
# routinely code-switch mid-sentence — "mixed" captures that, not just a stray
# leftover English word.
#   he:    pure Hebrew
#   en:    pure English
#   mixed: code-switched Hebrew/English in the same message
LANGUAGE_MODES: tuple[str, ...] = ("he", "en", "mixed")

# The exact JSON shape the model is trained to produce. Keep this in sync
# with whatever your Haiku classifier in services/ai returns, so swapping
# tiers doesn't require changing the confirm/dispatch code downstream.
OUTPUT_SCHEMA_DESCRIPTION = """\
Respond with ONLY a JSON object, no prose, no markdown fences:
{
  "intent": "<one of: %s>",
  "confidence": <float 0-1>,
  "slots": { "<slot_name>": "<value>", ... }
}
""" % ", ".join(INTENT_NAMES)


def slots_for(intent_name: str) -> tuple[str, ...]:
    spec = INTENTS[intent_name]
    return spec.required_slots + spec.optional_slots
