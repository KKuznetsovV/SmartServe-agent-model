"""
LoRA fine-tune a small open model on the SmartServe intent-extraction task.

Uses Unsloth for speed and lower VRAM usage on a single local GPU. Start
with the 1.5B model to get the full pipeline working end to end quickly —
data loading, prompt formatting, training loop, checkpoint saving — before
spending time on a bigger model. Swap MODEL_NAME to try Qwen2.5-7B-Instruct
once you have headroom and a working baseline to compare against.

Run:
    python training/train.py --config training/config.yaml

Requires a CUDA GPU. See training/requirements.txt for the environment —
keep this in its own venv, separate from SmartServe's Python services.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset


def _patch_datasets_dill_for_py314() -> None:
    """Work around a `datasets` + Python 3.14 incompatibility.

    `datasets.utils._dill.Pickler._batch_setitems` was written against the
    pre-3.14 `pickle.Pickler._batch_setitems(self, items)` signature. Python
    3.14's `pickle.save_dict` now calls `self._batch_setitems(items, obj)`
    (3 args), so any fingerprinting call (used by `load_dataset` and
    `.map()`) crashes with:
        TypeError: _Pickler._batch_setitems() takes 2 positional
        arguments but 3 were given
    This re-implements the same method, just forwarding the extra `obj`
    arg through to dill's underlying call instead of dropping it. Safe to
    remove once `datasets` ships an upstream Python 3.14 fix.
    """
    import sys

    if sys.version_info < (3, 14):
        return

    import datasets.utils._dill as _dp
    import dill

    def _patched_batch_setitems(self, items, obj=None):
        if self._legacy_no_dict_keys_sorting:
            return dill.Pickler._batch_setitems(self, items, obj)
        try:
            items = sorted(items)
        except Exception:  # noqa: BLE001 - matches upstream's broad catch
            from datasets.fingerprint import Hasher

            items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        dill.Pickler._batch_setitems(self, items, obj)

    _dp.Pickler._batch_setitems = _patched_batch_setitems


def _patch_triton_for_mingw_windows() -> None:
    """Work around triton-windows gaps when building with gcc/mingw instead
    of MSVC. None of this is needed if `CC` points at `cl.exe`.

    triton bundles its own mini CUDA Toolkit (ptxas, cuda.h, cuda.lib) under
    `triton/backends/nvidia/`, used to build a small native `cuda_utils`
    driver glue module the first time a kernel runs. Three gaps, uncovered
    by actually attempting the build (not by reading changelogs):

    1. `find_cuda_bundled()` looks for that bundled CUDA under
       `sysconfig.get_paths()["platlib"]`, which assumes packages live in
       the system site-packages dir. If they were installed to the
       per-user site-packages instead (no venv), it looks in the wrong
       place and `find_cuda()` reports "Failed to find CUDA" even though
       everything needed is actually present. Fix: point `CUDA_PATH` at
       triton's real install location so `find_cuda_env()` (checked first)
       finds it directly.
    2. `triton.runtime.build.normalize_path()` always prefixes paths with
       Windows' long-path escape (two backslashes + question mark +
       backslash). `_build()` only strips it back off (via an 8.3
       short-path conversion) when the compiler is MSVC, so gcc's `cc1`
       receives a literal prefixed path it doesn't understand and fails
       with "fatal error: cuda_utils.c: No such file or directory". Fix:
       skip the long-path prefix for non-MSVC compilers (safe here since
       build temp dirs are well under the 260-char limit).
    3. `triton.runtime.build._library_flag()` emits bare `cuda.lib` /
       `python314.lib` (correct for MSVC's `link.exe`, which applies
       `/LIBPATH` to bare names) but GNU `ld` does NOT apply `-L` search
       dirs to bare positional filenames, only to `-l<name>`/`-l:<name>`
       flags, so linking fails with "cannot find cuda.lib". Fix: use the
       `-l:<exact filename>` form for non-MSVC compilers on Windows.
    """
    import os
    import sys

    if sys.platform != "win32":
        return

    from triton.runtime import build as _tb

    cc = _tb.get_cc()
    if _tb.is_msvc(cc) or _tb.is_clang_cl(cc):
        return  # MSVC already has real long-path + bare-.lib handling

    # 1. Point CUDA_PATH at triton's own bundled mini CUDA Toolkit.
    if not os.getenv("CUDA_PATH") and not os.getenv("CUDA_HOME"):
        triton_pkg_dir = os.path.dirname(os.path.dirname(_tb.__file__))  # .../triton
        bundled_cuda = os.path.normpath(os.path.join(triton_pkg_dir, "backends", "nvidia"))
        if os.path.exists(os.path.join(bundled_cuda, "lib", "x64", "cuda.lib")):
            os.environ["CUDA_PATH"] = bundled_cuda

    # 2. Don't UNC-prefix paths for non-MSVC compilers.
    def _normalize_path_no_unc_prefix(path: str) -> str:
        return os.path.abspath(path).replace("/", "\\")

    _tb.normalize_path = _normalize_path_no_unc_prefix

    # 3. Use GNU ld's exact-filename search syntax instead of bare names.
    _orig_library_flag = _tb._library_flag

    def _patched_library_flag(lib: str, cc: str) -> str:
        if os.name == "nt" and not _tb.is_tcc(cc) and not (_tb.is_msvc(cc) or _tb.is_clang_cl(cc)):
            if not lib.lower().endswith(".lib"):
                lib = lib + ".lib"
            return f"-l:{lib}"
        return _orig_library_flag(lib, cc)

    _tb._library_flag = _patched_library_flag


PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def format_prompts(example: dict) -> dict:
    return {
        "text": PROMPT_TEMPLATE.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    parser.add_argument("--max-steps", type=int, default=None, help="Cap total training steps (for smoke tests); overrides num_epochs")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override config's output_dir (e.g. to avoid clobbering real checkpoints during a smoke test)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = str(args.output_dir) if args.output_dir else cfg["output_dir"]

    _patch_datasets_dill_for_py314()

    # Imported here, not at module level, so `--help` and config parsing
    # work even before the (heavy, GPU-only) unsloth/torch stack is set up.
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    _patch_triton_for_mingw_windows()

    print(f"Loading base model: {cfg['model_name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,  # auto-detect
        load_in_4bit=cfg["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["train_path"],
            "validation": cfg["val_path"],
        },
    )
    dataset = dataset.map(format_prompts)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=SFTConfig(
            dataset_text_field="text",
            max_length=cfg["max_seq_length"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            warmup_steps=cfg["warmup_steps"],
            num_train_epochs=cfg["num_epochs"],
            max_steps=args.max_steps if args.max_steps else -1,
            learning_rate=cfg["learning_rate"],
            fp16=not cfg["load_in_4bit"],
            bf16=cfg["load_in_4bit"],
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=cfg["eval_steps"],
            save_strategy="steps",
            save_steps=cfg["eval_steps"],
            output_dir=output_dir,
            seed=cfg["seed"],
            report_to="none",
        ),
    )

    print("Starting training...")
    trainer.train()

    final_dir = Path(output_dir) / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved fine-tuned adapter to {final_dir}")
    print("Next: python inference/quantize.py to export a GGUF for serving.")


if __name__ == "__main__":
    main()
