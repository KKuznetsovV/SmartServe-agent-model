# SmartServe Local Agent Model

A fine-tuned, locally-hosted small language model that classifies restaurant
manager requests (e.g. *"raise the price of the margherita by 2 shekels"*,
*"route bar tickets to the kitchen printer"*) into structured actions for
[SmartServe](../SmartServe), an AI-powered restaurant POS system, without a
network call to a cloud LLM API.

Final project for the Full Stack Development course (John Bryce).

## Why

SmartServe already uses Claude (Haiku for classification, Sonnet for
complex reasoning) to let non-technical managers configure the system in
natural language. Every request, however trivial, costs an API call. This
project asks: for the common, structured requests (price changes, printer
routing, simple item edits), can a small model running entirely locally
handle them, instantly and for free, falling back to the cloud pipeline
only when it's not confident?

## Approach

1. **Synthetic data generation.** No real usage data exists yet (the app is
   still in demo stage), so training data is bootstrapped using a
   Self-Instruct-style approach: a larger "teacher" model (Claude Sonnet)
   generates diverse, realistic manager phrasings against many different
   synthetic restaurant menus, each labeled with the correct structured
   action. See [`data/`](data/).
2. **Fine-tuning.** A small open model (Qwen2.5-1.5B-Instruct, see
   [`training/config.yaml`](training/config.yaml)) is fine-tuned with LoRA
   using [Unsloth](https://github.com/unslothai/unsloth) for fast local
   training on a single GPU. See [`training/`](training/).
3. **Evaluation against a held-out set of unseen menus** (not just a
   train/test split of the same distribution — entire restaurants the model
   never saw), compared directly against Claude Haiku's accuracy on the
   same test set. This is the key result: does the model generalize to new
   menus, or did it just memorize the training set? See
   [`evaluation/`](evaluation/) and [`evaluation/results/`](evaluation/results/).
4. **Deployment as a first tier**, quantized to GGUF and slotted in front of
   SmartServe's existing Haiku/Sonnet classification pipeline: local model
   answers when confident, falls through to the existing cloud pipeline
   otherwise. See [`docs/integration.md`](docs/integration.md).

## Repo structure

```
data/         synthetic menu + labeled example generation
training/     LoRA fine-tuning (Unsloth)
evaluation/   held-out generalization test vs. Claude Haiku baseline
inference/    GGUF export + a standalone test server
docs/         integration guide for wiring this back into SmartServe
common/       shared intent schema used by all of the above
```

## Quickstart

```bash
# 1. Generate synthetic menus
python data/generate_menus.py --count 40 --out data/raw/menus.jsonl

# 2. Generate labeled training examples (calls the Claude API — set ANTHROPIC_API_KEY)
python data/generate_dataset.py --menus data/raw/menus.jsonl --per-intent-per-menu 6

# 3. Split into train/val
python data/split_dataset.py

# 4. Fine-tune (requires a CUDA GPU — see training/requirements.txt)
python training/train.py --config training/config.yaml

# 5. Evaluate against held-out menus, vs. Claude Haiku
python evaluation/evaluate.py --model-path training/checkpoints/final

# 6. Quantize for lightweight serving
python inference/quantize.py --model-path training/checkpoints/final

# 7. Try it locally
python inference/serve.py --model inference/model.gguf
```

## Results

_Fill in after running `evaluation/evaluate.py` — that script writes
`evaluation/results/report.json` with accuracy and latency numbers. Pull
the summary into a short table here, e.g.:_

| | Accuracy (held-out menus) | Avg. latency |
|---|---|---|
| Claude Haiku (API) | — | — |
| Local model (this project) | — | — |

## Intents covered

See [`common/schema.py`](common/schema.py) for the full, authoritative
list and slot definitions: `update_price`, `create_item`, `delete_item`,
`update_description`, `update_item_name`, `route_printer`, `update_stock`,
plus an `unclear` fallback intent that deliberately triggers escalation
rather than a wrong guess.

## Scope and limitations

- Trained on synthetic, LLM-generated data, not real production usage
  (the SmartServe app has no real clients yet). Accuracy on genuine manager
  phrasing in production may differ; the audit log SmartServe already
  collects would be the natural source for a v2 fine-tune on real data.
- Handles single-turn, single-intent requests only, matching SmartServe's
  current `/ai/command` design. Multi-step or ambiguous requests are meant
  to fall through to the existing cloud pipeline, not be handled locally.
- This repo trains and evaluates the model. It does not modify the
  SmartServe application — see `docs/integration.md` for that separate,
  smaller step.
