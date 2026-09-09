"""FastAPI service for Qwen 2.5 Coder inline completions.

Run locally with:
    python autocomplete_service.py --model-path /path/to/Qwen2.5-Coder

The browser client posts ``{"text": "..."}`` to ``/autocomplete`` and
receives the generated suffix in the ``suggestion`` field.
"""

import argparse
import os
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = os.getenv("AUTOCOMPLETE_MODEL_PATH", "/models/qwen2.5-coder")
MAX_INPUT_CHARS = 4000
DEFAULT_MAX_NEW_TOKENS = 100


model = None
tokenizer = None
device = "uninitialized"


class AutocompleteRequest(BaseModel):
    # ``text`` is the document prefix up to the cursor. The browser also
    # sends the suffix for future infill-aware model support.
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    cursor_position: int | None = Field(default=None, ge=0)
    suffix: str = Field(default="", max_length=MAX_INPUT_CHARS)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=256)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)


class AutocompleteResponse(BaseModel):
    suggestion: str
    device: str
    model: str


def get_device() -> str:
    target = os.getenv("TARGET_DEVICE", "").lower()
    if target:
        if target not in {"cpu", "cuda"}:
            raise RuntimeError("TARGET_DEVICE must be either 'cpu' or 'cuda'.")
        return target
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_device_details() -> str:
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU" if device == "cpu" else "uninitialized"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device

    device = get_device()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TARGET_DEVICE=cuda was requested, but CUDA is unavailable.")

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32
    )
    print(f"Loading Qwen 2.5 Coder on '{device}' (dtype: {dtype})...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print("Autocomplete model loaded.")

    yield

    model = None
    tokenizer = None
    if device == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(
    title="Autocomplete Engine API",
    description="Inline code completion powered by Qwen 2.5 Coder.",
    version="1.0.0",
    lifespan=lifespan,
)

# Set AUTOCOMPLETE_ALLOWED_ORIGINS to a comma-separated list in production.
allowed_origins = [
    origin.strip()
    for origin in os.getenv("AUTOCOMPLETE_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials="*" not in allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def generate_completion(request: AutocompleteRequest) -> str:
    if model is None or tokenizer is None:
        raise RuntimeError("The autocomplete model is not initialized.")

    # Qwen2.5-Coder supports fill-in-the-middle prompts. Including the
    # suffix helps the model choose a completion that fits the code already
    # present after the cursor, rather than blindly continuing the prefix.
    prefix = request.text[-MAX_INPUT_CHARS:]
    suffix = request.suffix[:MAX_INPUT_CHARS]
    if suffix:
        prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    else:
        prompt = prefix

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[-1]

    generation_kwargs = {
        **inputs,
        "max_new_tokens": request.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": request.temperature > 0,
        "top_p": request.top_p,
    }
    if request.temperature > 0:
        generation_kwargs["temperature"] = request.temperature

    with torch.inference_mode():
        output = model.generate(**generation_kwargs)

    generated_tokens = output[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).rstrip()


@app.post("/autocomplete", response_model=AutocompleteResponse)
def autocomplete(request: AutocompleteRequest):
    try:
        suggestion = generate_completion(request)
        return AutocompleteResponse(
            suggestion=suggestion,
            device=device,
            model=os.path.basename(MODEL_PATH.rstrip("/")),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": device,
        "device_details": get_device_details(),
    }


@app.get("/ready")
def ready():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Autocomplete service is not ready.")
    return {"status": "ready", "device": device}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen 2.5 Coder autocomplete FastAPI server")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Local Qwen model directory")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8004")), help="Port number")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cuda", action="store_true", help="Force CUDA")
    group.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    if args.cuda:
        os.environ["TARGET_DEVICE"] = "cuda"
    elif args.cpu:
        os.environ["TARGET_DEVICE"] = "cpu"

    uvicorn.run(app, host=args.host, port=args.port)
