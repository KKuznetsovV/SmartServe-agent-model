"""
Export the fine-tuned model to a quantized GGUF file for lightweight
serving (via llama.cpp / llama-cpp-python), rather than loading the full
PyTorch + Transformers stack in production.

Run:
    python inference/quantize.py --model-path training/checkpoints/final \\
        --out inference/model.gguf --quant q4_k_m

q4_k_m is a good default: a strong accuracy/size tradeoff for a 1.5B-7B
model. Try q8_0 first if you want to sanity-check that quantization isn't
what's hurting accuracy, then drop down once you've confirmed q4 is fine.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("training/checkpoints/final"))
    parser.add_argument("--out", type=Path, default=Path("inference/model.gguf"))
    parser.add_argument("--quant", default="q4_k_m", choices=["q4_k_m", "q5_k_m", "q8_0"])
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    print(f"Loading {args.model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(str(args.model_path))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to GGUF ({args.quant})... this can take a few minutes.")
    model.save_pretrained_gguf(
        str(args.out.parent),
        tokenizer,
        quantization_method=args.quant,
    )
    print(f"Done. Look for the .gguf file in {args.out.parent}")
    print("Next: python inference/serve.py --model <path-to>.gguf to test it locally.")


if __name__ == "__main__":
    main()
