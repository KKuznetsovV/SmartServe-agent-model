# SmartServe Local Agent Model

A fine-tuned, locally-hosted small language model that classifies restaurant
manager requests (e.g. *"raise the price of the margherita by 2 shekels"*,
*"route bar tickets to the kitchen printer"*) into structured actions for
[SmartServe](../SmartServe), an AI-powered restaurant POS system, without a
network call to a cloud LLM API.


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
   - **Hebrew-primary by design, not as an afterthought.** SmartServe is
     built for the Israeli restaurant market, where Hebrew is the language
     managers actually type in day to day (see the SmartServe repo's own
     `docs/architecture.md` — "RTL-native", "Hebrew is a first-class
     language"). Israeli businesses also routinely code-switch mid-sentence
     (e.g. *"תעדכן את המחיר של ה-Margherita ל-55 שקל"*), so the generator
     produces a controlled mix rather than leaving language to chance:
     ~50% pure Hebrew, ~20% pure English, ~30% realistic code-switched
     Hebrew/English, configurable via `--hebrew-fraction` /
     `--codeswitch-fraction`. Every example is tagged with its
     `language_mode` (`he` / `en` / `mixed`) end to end, so accuracy can be
     (and is) measured per-language rather than hidden behind one aggregate
     number — see Results below.
2. **Fine-tuning.** A small open model (Qwen2.5-1.5B-Instruct, see
   [`training/config.yaml`](training/config.yaml)) is fine-tuned with LoRA
   using [Unsloth](https://github.com/unslothai/unsloth) for fast local
   training on a single GPU. See [`training/`](training/) and "Base model
   choice" below for why this is worth reconsidering given the Hebrew focus.
3. **Evaluation against a held-out set of unseen menus** (not just a
   train/test split of the same distribution — entire restaurants the model
   never saw), compared directly against Claude Haiku's accuracy on the
   same test set, broken down by `language_mode`. This is the key result:
   does the model generalize to new menus (and to Hebrew as well as
   English), or did it just memorize the training set? See
   [`evaluation/`](evaluation/) and [`evaluation/results/`](evaluation/results/).
4. **Deployment as a first tier**, quantized to GGUF and slotted in front of
   SmartServe's existing Haiku/Sonnet classification pipeline: local model
   answers when confident, falls through to the existing cloud pipeline
   otherwise. See [`docs/integration.md`](docs/integration.md).

### Base model choice: is Qwen2.5-1.5B-Instruct actually a good Hebrew choice?

Worth being upfront about, since it's a real tradeoff and not an obvious
default given the Hebrew-primary decision above:

- **Tokenizer efficiency.** Qwen2.5's own model card lists ~29 supported
  languages by name (Chinese, English, French, Spanish, Portuguese, German,
  Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more) —
  Hebrew is not among them, unlike the closely-related Arabic abjad script.
  Its BPE tokenizer will still tokenize Hebrew (byte-fallback guarantees
  that), but likely fragments Hebrew words into far more, shorter
  sub-word/byte tokens than a Hebrew-aware tokenizer would — a Hebrew-poor
  tokenizer silently eats more of the (small, 1024-token) context window per
  message and gives the model less signal per fine-tuning example to learn
  from.
- **DictaLM 2.0** ([`dicta-il/dictalm2.0-instruct`](https://huggingface.co/dicta-il/dictalm2.0-instruct),
  Apache-2.0), from DICTA (Israel's Center for Text Analysis), exists
  specifically to fix this — its own paper is literally titled *"Adapting
  LLMs to Hebrew: Unveiling DictaLM 2.0 with Enhanced Vocabulary"*
  ([arXiv:2407.07080](https://arxiv.org/abs/2407.07080)), built on
  Mistral-7B with a Hebrew-extended vocabulary for meaningfully better
  tokens-per-character on Hebrew text.
- **The catch:** DictaLM 2.0 is 7B parameters — ~4.6x the current 1.5B
  default — which cuts against the "instant, free, local" first-tier design
  goal (more VRAM, higher latency, bigger GGUF to ship). It also doesn't
  have the same turnkey Unsloth support: Unsloth's model hub only hosts
  `unsloth/DictaLM-3.0-24B-Thinking-GGUF` (a much larger, unrelated
  reasoning model, GGUF-only), not a pre-quantized `bnb-4bit` LoRA-ready
  repo the way `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` is. DictaLM 2.0's
  Mistral architecture is still Unsloth-supported generically, so
  `load_in_4bit=True` against the vanilla `dicta-il/dictalm2.0-instruct`
  checkpoint should work — just without a pre-baked quant, so the first
  load quantizes on the fly instead of downloading ready-made 4-bit weights.

**Recommendation:** don't switch blindly. Keep Qwen2.5-1.5B-Instruct as the
running baseline (it's small, fast, and has the turnkey Unsloth quant), but
treat DictaLM 2.0 as a real contender worth actually measuring given Hebrew
is the primary language here — `training/config.yaml`'s own comment already
anticipates exactly this kind of side-by-side ("duplicate this file... to
try the larger model and compare results"). A quick, cheap first check
before committing to a full DictaLM fine-tune: tokenize a sample of
generated Hebrew messages with both tokenizers and compare tokens-per-
character directly — that's a fast way to confirm the fragmentation problem
above is real before paying for a 7B fine-tune to fix it. This is a decision
for you to make, not one this pass makes for you.

## Repo structure

```
data/         synthetic menu + labeled example generation
training/     LoRA fine-tuning (Unsloth)
evaluation/   held-out generalization test vs. Claude Haiku baseline
inference/    GGUF export + a standalone test server
docs/         integration guide + Windows-specific build/setup notes
common/       shared intent schema used by all of the above
```

## Quickstart

```bash
# 1. Generate synthetic menus
python data/generate_menus.py --count 40 --out data/raw/menus.jsonl

# 2. Generate labeled training examples (calls the Claude API — set ANTHROPIC_API_KEY)
#    Defaults to ~50% Hebrew / ~30% code-switched / ~20% English; override with
#    --hebrew-fraction / --codeswitch-fraction if you want a different mix.
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

## Every number below is measured, not estimated

It would be easy to read the table below as "fine-tune a model, get a clean
result." That's not what happened, and it's worth being explicit about it
rather than letting a clean-looking final table imply a clean process. At
least four times, a result that looked wrong on its face got caught and
root-caused before it shipped into this README, not reported as-is:

- **Haiku's first "accuracy" was 12.5%** — implausible for a frontier
  model on simple classification. Root cause: Haiku wraps its JSON in
  ` ```json...``` ` fences despite being told not to, and
  `evaluate.py`'s parser didn't strip them, so every real-intent call
  silently fell back to `"unclear"` — the bogus 12.5% figure exactly
  matched the fraction of genuinely-`unclear` test examples, which is
  what made it suspicious enough to investigate rather than report. Fixed
  in `_parse_json_response()` (applied to both Haiku and local-model
  parsing) before the real 98.3% number above was ever generated.
- **Three data-generation intents were emitting hardcoded, static slot
  values** instead of teacher-invented ones (`update_description`,
  `create_item`, `update_item_name`) — a bug that would have taught the
  fine-tuned model to reproduce a fixed placeholder string rather than
  learn to extract/invent slot values from context. Caught by a
  small-batch spot-check before scaling up generation, not after.
- **Message opening-phrases converged hard onto 1-2 templates** for the
  short, structurally simple intents — one Hebrew opening alone covered
  72% of `update_price` examples in an early batch. Fixed via
  `PHRASING_STYLE_DIRECTIVES` (verified 13/80 → 57/80 unique openings on
  `update_price`), though a partial regression resurfaced at 4x the data
  volume — documented honestly as an accepted, measured limitation rather
  than re-chased indefinitely (see "Known limitations" below), and
  directly answered with the template-vs-long-tail accuracy breakdown
  further down rather than left as an open worry.
- **Fixing the phrasing convergence introduced a JSON reliability
  regression** — malformed-JSON skips on `update_description` ticked up
  from 0% to 3.75% right after the phrasing-style directive was added.
  Root-caused to prompt ordering (the phrasing instruction became the
  last, most recent thing the teacher read, making the earlier JSON-only
  contract comparatively "stale"), fixed by moving a JSON-output reminder
  to the end of the prompt, and verified back to 0% before moving on.

None of these were caught by a test suite or a linter — they surfaced by
treating an implausibly good *or* implausibly bad number as a signal to go
read the raw examples and logs, not something to write into a results
table as-is. That standard applies to every number below, including the
GPU latency figure, which was independently confirmed via load-time logs
and `nvidia-smi` utilization (see "Quantization" below) rather than
accepted on the strength of a successful build.

## Results

Real evaluation run: 3-epoch fine-tune of `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`
(690 steps, final `eval_loss` 0.1747 on the training-time validation split),
scored against **9 synthetic restaurant menus held out entirely from
training data generation** (see `data/generate_menus.py` /
`data/generate_dataset.py`'s `--holdout-fraction 0.15` split — verified by
direct inspection that zero training examples reference any held-out
`business_id`). 360 held-out test messages (5 per intent per menu, 8
intents), scored with `evaluation/evaluate.py --n-per-intent 5`. Full
machine-readable results: `evaluation/results/report.json`.

| | Accuracy (held-out menus) | Avg. latency |
|---|---|---|
| Claude Haiku (API, `claude-haiku-4-5`) | 98.3% (354/360) | API round-trip (not directly comparable to local; see below) |
| Local model, unquantized | 99.2% (357/360) | 5,788 ms/example (RTX 5060 Laptop, 8GB, 4-bit, unbatched `generate()`) |
| Local model, quantized (`q4_k_m` GGUF, **CPU-only**) | 99.4% (358/360) | 2,670 ms/example (see caveat below — not the deployability number) |
| Local model, quantized (`q4_k_m` GGUF, **GPU-accelerated**) | 99.2% (357/360) | **264 ms/example** (RTX 5060 Laptop, all 28 layers offloaded to CUDA) |

The local model matches Haiku overall — the ~1 point gap (357/360 vs.
354/360, a difference of 3 examples) is within normal sampling noise at
n=360 and shouldn't be read as the local model "beating" Haiku, just as a
genuinely credible tie for a 1.5B 4-bit model against a frontier API model
on this narrow, well-specified task. The result worth highlighting is the
per-language breakdown below, not the aggregate number. The CPU-only
quantized latency (2,670ms) was a real but environment-constrained
measurement, not the deployability figure — the GPU-accelerated quantized
run (264ms/example) is the number that actually matters and clears the
sub-1s interactive-use target comfortably (see "Quantization" below).

### Quantization

`inference/quantize.py --quant q4_k_m` was run against `training/checkpoints/final`,
producing a 940MB GGUF (`inference_gguf/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf`).
It was re-evaluated against **the identical 360 held-out examples** used
above (loaded directly from `evaluation/results/report.json`'s
`per_example` list — no test set regeneration, no Haiku re-scoring; see
`evaluation/evaluate_quantized.py`), so the accuracy comparison is
apples-to-apples.

- **Accuracy: no meaningful loss from quantization.** 99.4% (358/360),
  actually one example better than unquantized — within noise, same
  conclusion as the Haiku comparison above (don't read this as
  quantization "improving" the model). The 2 quantized misses are the same
  failure *pattern* as the unquantized run (`delete_item` on long-tail
  phrasing), not a new quantization-induced weakness — one of the
  unquantized run's 3 misses (the `תוריד`/מוחיטו ambiguity) is actually
  fixed under quantization, another new one appears elsewhere; net effect
  is a wash, not a regression.
- **Latency (CPU-only): real speedup over unquantized-GPU, but not the
  deployability number.** 2,670 ms/example average (median 2,494ms, range
  1,100–5,421ms across the 360 examples) — under half the unquantized
  5,788ms figure, but that comparison mixes hardware (CPU vs GPU) and
  isn't apples-to-apples. This was a real, environment-constrained
  measurement at the time (`llama-cpp-python` had no CUDA Toolkit
  available to build against), not a deployability estimate. Thread count
  was tuned (a sweep found `n_threads=4` fastest — this workload is
  memory-bandwidth bound, and the default of all 28 logical cores was ~2x
  *slower*, not faster).
- **Latency (GPU-accelerated): the real deployability number.** CUDA
  Toolkit 13.1 was installed and `llama-cpp-python` rebuilt from source
  with `-DGGML_CUDA=on` (Ninja generator + `-allow-unsupported-compiler`,
  since the installed MSVC toolset is newer than nvcc 13.1's officially
  supported range — verified to still produce correct, working kernels).
  This build hit three distinct, genuine Windows-specific blockers (the
  Ninja-generator fix above, the compiler-version flag above, and a DLL
  search-path issue), on top of three more hit earlier during training —
  six real environment problems root-caused across this project, all
  written up in [`docs/windows-setup.md`](docs/windows-setup.md) since a
  future Windows user would hit the same walls.
  Re-running the identical 360-example benchmark with `n_gpu_layers=-1`
  gives **264.4 ms/example average** (median 243.5ms, range 126.1–593.9ms)
  — roughly **10x faster than the CPU-only build** and **~22x faster than
  the unquantized GPU baseline**. GPU use was confirmed two ways, not just
  assumed from a successful compile: llama.cpp's own load-time logs show
  all 28 layers assigned `dev = CUDA0` with a dedicated CUDA compute
  buffer, and `nvidia-smi` showed 76-89% GPU utilization while the
  benchmark was actually running. Accuracy on this run was 99.2%
  (357/360, he 99.5%/en 96.8%/mixed 100.0%) — a 1-example difference from
  the CPU quantized run (358/360), consistent with expected
  floating-point non-determinism between CPU and CUDA reduction kernels,
  not a real regression (same `delete_item`/long-tail failure pattern).
- **`q8_0` was not tried.** The task was conditional on `q4_k_m` costing
  meaningful accuracy — it didn't (see above), so there's no
  accuracy/speed tradeoff to rebalance, and `q8_0` uses more bits per
  weight than `q4_k_m` so it would only make latency *worse*, working
  against the actual goal here. Not silently skipped: this is the
  reasoned call given the conditional trigger wasn't met.

**Recommendation: ready to deploy on latency grounds (GPU-accelerated).**
The target was "comfortably under a second, ideally well under that" —
the GPU-accelerated quantized build clears this easily, with even its
slowest of the 360 calls (594ms) staying under the 1s bar, and the 264ms
average leaving comfortable headroom for real-world overhead (request
queuing, prompt construction, etc.). The earlier CPU-only figure (2,670ms)
was a real but hardware-constrained measurement, not a reflection of the
model's actual deployability — with a CUDA-enabled build on the target
GPU, both the accuracy question (no meaningful loss from quantization) and
the speed question (well under the 1s target) are resolved.

**By language** — a single aggregate number can hide a model that's strong
in English and much weaker in Hebrew, which would be a real problem for an
Israeli-market product, so this breakdown is the more important table for
this project specifically:

| language_mode | Haiku accuracy | Local accuracy (unquantized) | Local accuracy (`q4_k_m`) | n |
|---|---|---|---|---|
| `he` (pure Hebrew) | 98.4% | 98.4% | 99.5% | 193 |
| `en` (pure English) | 100.0% | 100.0% | 98.4% | 63 |
| `mixed` (code-switched) | 97.1% | 100.0% | 100.0% | 104 |

No language-specific weakness for the local model — if anything it's
slightly stronger than Haiku on code-switched Hebrew/English input, which
is the input pattern SmartServe's real managers are expected to use most.
Quantization doesn't change this picture (the `en` dip to 98.4%, 62/63, is
a single example — noise at this n, not a real regression).

**By phrasing stratum (template_match vs long_tail)** — this directly
answers the open question from the data-generation phase about whether the
residual opening-phrase clustering (10-14% dominance on a few intents, see
"Known limitations" below) hurt generalization:

| intent | haiku (template) | haiku (long-tail) | local (template) | local (long-tail) | n (template) | n (long-tail) |
|---|---|---|---|---|---|---|
| `update_price` | 100.0% | 100.0% | 100.0% | 100.0% | 16 | 29 |
| `create_item` | 100.0% | 100.0% | 100.0% | 100.0% | 15 | 30 |
| `delete_item` | n/a | 100.0% | n/a | 95.6% | 0 | 45 |
| `update_description` | 100.0% | 100.0% | 100.0% | 97.1% | 11 | 34 |
| `update_item_name` | 100.0% | 100.0% | 100.0% | 100.0% | 9 | 36 |
| `route_printer` | 100.0% | 100.0% | 100.0% | 100.0% | 2 | 43 |
| `update_stock` | 100.0% | 93.0% | 100.0% | 100.0% | 2 | 43 |
| `unclear` | 90.6% | 100.0% | 100.0% | 100.0% | 32 | 13 |

**Verdict: the clustering turned out to be benign, not a real gap.** The
largest local template-vs-long-tail delta is 4.4 points (`delete_item`,
95.6% vs a template stratum with zero examples to compare against — not a
meaningful signal), and 2.9 points on `update_description`. Nothing close
to the >15-point threshold `evaluate.py` flags as a real generalization
concern. The model is not overfitting to the dominant training phrasings.

**Local model failure cases** (3 of 360 total; all Hebrew, all `long_tail`):

- `היי, תוריד לי את המוחיטו מההזמנה בבקשה 🙏` → expected `delete_item`,
  predicted `update_price`. This is a genuine linguistic ambiguity, not a
  random miss: **תוריד** literally means "lower/take down", and is used
  both for "lower the price" and "take the item down (remove it)" in
  colloquial Hebrew — Haiku got this one right, so it's a real (if narrow)
  local-model weakness worth more Hebrew delete-item training examples
  using this exact verb, not a data-quality bug.
- `היי, תוציאו את האספרסו מרטיני מהתפריט בבקשה...` → expected `delete_item`,
  predicted `unclear` (safe fallback, not a wrong guess).
- Long, multi-clause `update_description` message → expected
  `update_description`, predicted `unclear` (safe fallback).

Two of the three misses fall back to `unclear` (the safe, escalate-to-human
behavior SmartServe wants when uncertain) rather than a confidently wrong
answer — only the `מוחיטו`/מוריד example above is a genuinely confident
wrong prediction. (The CPU-quantized model's 2 misses and the
GPU-quantized model's 3 misses are each a different subset of
`delete_item`/long-tail examples — same failure *pattern* across all four
runs (unquantized, CPU-quantized, GPU-quantized), not a new weakness
introduced by quantization or by CUDA vs CPU floating-point differences;
see "Quantization" above.)

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

### Known limitations in the generated training data

Surfaced by `data/spot_check.py --check-repetition` on the full 4,078-example
generation run. Both are documented, accepted tradeoffs rather than bugs —
kept as-is rather than chased further, for the reasons below:

- **Residual message opening-phrase clustering (10-14% on affected
  intents).** Even after the `PHRASING_STYLE_DIRECTIVES` fix (which took the
  worst offenders from 49-91% dominance down to this range), a handful of
  very natural, idiomatic openings (e.g. Hebrew *"אפשר לעדכן את המחיר
  של..."*) still recur across multiple phrasing-style directives instead of
  being confined to one. Not chased further because: (a) these are
  genuinely realistic phrasings a manager would actually type, not garbled
  or repetitive-looking text — the risk is the model over-anchoring on
  them, not that they degrade data quality on their face; (b) diminishing
  returns on further prompt-tuning after two already-verified passes;
  (c) `evaluation/evaluate.py` now measures the actual question this
  raises directly (template_match vs long_tail accuracy, see Results above)
  instead of guessing — if it turns out to be a real generalization gap,
  that's the signal to invest more here, not a blind further pass now.
- **`update_item_name`'s invented `new_name` values are less diverse than
  the other creative slots (61.8% unique vs. ~99% for `update_description`
  and `create_item`'s descriptions).** Attributed to a structural cause,
  not a generation bug: renamed values naturally cluster around the fixed
  pool of ~35 base item names repeated across only 7 cuisines over 60
  synthetic menus, so the same starting point (e.g. "Margherita Pizza")
  recurs often and its plausible renamings are a smaller space than a
  free-text description. Fixing this would mean generating a larger/more
  varied base menu-item catalog (a `data/generate_menus.py` change), not a
  `generate_dataset.py` prompt tweak — out of scope for this pass.
