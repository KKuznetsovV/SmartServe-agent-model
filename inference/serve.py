"""
Minimal local test server for the quantized model — use this to sanity
check inference before wiring it into SmartServe's services/ai.

Run:
    pip install llama-cpp-python fastapi uvicorn
    python inference/serve.py --model inference/model.gguf

Then:
    curl -X POST localhost:8090/classify \\
        -H "Content-Type: application/json" \\
        -d '{"message": "raise the price of the margherita by 2 shekels"}'

See docs/integration.md for how this logic gets embedded into
SmartServe's services/ai as an actual first-tier check, rather than run as
a separate standalone service.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.schema import OUTPUT_SCHEMA_DESCRIPTION  # noqa: E402

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="SmartServe Local Agent — test server")
_llm = None  # loaded in main()

INSTRUCTION = (
    "You classify a restaurant manager's message into a structured action "
    "for a POS system. " + OUTPUT_SCHEMA_DESCRIPTION
)


class ClassifyRequest(BaseModel):
    message: str


class ClassifyResponse(BaseModel):
    intent: str
    confidence: float
    slots: dict
    source: str = "local_model"


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{req.message}\n\n### Response:\n"
    output = _llm(prompt, max_tokens=200, stop=["###"], temperature=0.0)
    text = output["choices"][0]["text"].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"intent": "unclear", "confidence": 0.0, "slots": {}}
    return ClassifyResponse(**parsed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("inference/model.gguf"))
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    global _llm
    from llama_cpp import Llama

    print(f"Loading {args.model}...")
    _llm = Llama(model_path=str(args.model), n_ctx=1024, verbose=False)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
