# Integrating into SmartServe

This repo produces a model. It does not modify SmartServe itself — keep
that boundary. This doc describes the (small) integration change you make
back in the `SmartServe` repo once the model is trained and evaluated.

## Where it plugs in

`services/ai`'s `/ai/command` endpoint currently does, roughly:

```
classify with Haiku
  -> if malformed / low confidence -> escalate to Sonnet
-> return preview for confirmation
```

The integration adds one tier in front, without changing the confirm/
dispatch logic at all:

```
classify with local model (fast, free, no network call)
  -> if confidence >= threshold and intent != "unclear" -> return preview
  -> else -> fall through to the existing Haiku/Sonnet pipeline, unchanged
```

This is intentionally a *narrow* change. The existing escalation path,
confirmation step, dispatch-to-catalog/printing logic, and audit logging
all stay exactly as they are. The local model only decides whether it's
confident enough to skip the API call — it never bypasses confirmation,
and it never talks to catalog/printing directly.

## Steps

1. **Package the model.** Push the quantized GGUF from `inference/` to a
   model registry — the Hugging Face Hub is the simplest free option for a
   solo/course project. Tag a release version so `services/ai` pins a
   specific version, not "whatever's newest."

2. **Add `llama-cpp-python` to `services/ai`'s dependencies** (it's a
   Python package, no extra infra needed — it runs the GGUF file directly
   in-process).

3. **Add a small `local_classifier.py` module in
   `services/ai/app/modules/ai/services/`**, mirroring the existing
   `anthropic_client.py`-style isolation the project already uses for
   Haiku/Sonnet. It should load the model once at service startup (not
   per-request) and expose a `classify(message: str) -> ClassificationResult`
   function with the *same* return shape the Haiku path already produces,
   so the router code downstream doesn't need to know which tier answered.

4. **Add a confidence threshold as a service setting** (env var, matching
   how `AI_INTENT_MODEL` etc. are already configured), so you can tune it
   without a redeploy. Start conservative (e.g. 0.9) — a wrong silent
   "confident" answer is worse than an unnecessary Haiku call.

5. **Update `router.py`'s command-handling to try the local classifier
   first**, falling through to the existing Haiku call on low confidence
   or an `unclear` result. The location-ownership validation that was
   already fixed to run at classify time should run before *either* tier,
   not be duplicated per-tier.

6. **Add a `source` field to the audit log** (`"local_model"` /
   `"haiku"` / `"sonnet"`), so you can see in production which tier
   actually handled each request — this is also useful evidence for your
   project writeup if you get any real usage before submission.

7. **Test with the existing pattern**: mock the local classifier the same
   way `anthropic_client` calls are already mocked in `services/ai`'s test
   suite, so CI doesn't need the actual model file.

## What NOT to do

- Don't have the local model call catalog/printing directly. It only
  classifies; dispatch stays exactly where it is today.
- Don't remove the Haiku/Sonnet path. The three-tier design (free instant
  local model → cheap fast Haiku → capable Sonnet) is the actual point —
  it's a stronger result than "we replaced the cloud model," which would
  just trade one single point of failure for another.
- Don't skip the confidence threshold and treat every local-model answer
  as final. The threshold is what keeps this safe to ship.
