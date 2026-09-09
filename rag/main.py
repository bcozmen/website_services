import argparse
from contextlib import asynccontextmanager
import os
import lancedb
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.getenv("RAG_MODEL_PATH", "/models/qwen3_embedding")
LANCEDB_PATH = os.getenv("RAG_LANCEDB_PATH", "/data/wiki_lancedb")
TABLE_NAME = os.getenv("RAG_TABLE_NAME", "wikipedia_rag")

INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


# ============================================================
# Global resources
# ============================================================

model = None
tbl = None


# ============================================================
# Helper function for device detection
# ============================================================

def get_device_info() -> dict:
    """Returns the current execution device status and details."""
    if model is None:
        return {"device": "uninitialized", "details": None}

    device_type = model.device.type
    details = (
        torch.cuda.get_device_name(model.device)
        if device_type == "cuda"
        else "CPU"
    )

    return {
        "device": device_type,  # 'cuda' or 'cpu'
        "details": details       # e.g., 'NVIDIA A100-SXM4-40GB' or 'CPU'
    }


# ============================================================
# Load model + database once when FastAPI starts
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tbl

    target_device = os.getenv("TARGET_DEVICE")
    if target_device:
        device = target_device.lower()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device not in {"cpu", "cuda"}:
        raise RuntimeError("TARGET_DEVICE must be either 'cpu' or 'cuda'.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    print(f"Loading embedding model on '{device}' (dtype: {dtype})...")

    model = SentenceTransformer(
        MODEL_PATH,
        device=device,
        model_kwargs={"torch_dtype": dtype} if device == "cuda" else {},
    )

    print("Connecting to LanceDB...")

    db = lancedb.connect(LANCEDB_PATH)
    tbl = db.open_table(TABLE_NAME)

    print("Model and database loaded.")

    yield

    model = None
    tbl = None


app = FastAPI(
    title="Wikipedia RAG API",
    description="Semantic search over a Wikipedia LanceDB index.",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Request / Response models
# ============================================================

class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    rank: int
    title: str
    url: str
    chunk_id: str | int | None = None
    chunk: str
    distance: float


class SearchResponse(BaseModel):
    question: str
    device: str  # <--- Added to convey device info in search results
    results: list[SearchResult]


# ============================================================
# Search
# ============================================================

def search_wikipedia(query_text: str, top_k: int = 5):
    if model is None or tbl is None:
        raise RuntimeError("Model or database is not initialized.")

    formatted_query = f"Instruct: {INSTRUCTION}\nQuery: {query_text}"

    with torch.inference_mode():
        query_vector = model.encode(
            formatted_query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    fetch_limit = max(top_k * 10, 50)

    results = (
        tbl.search(query_vector)
        .metric("cosine")
        .limit(fetch_limit)
        .select(["title", "text", "url", "chunk_id", "_distance"])
        .to_pandas()
    )

    unique_results = (
        results.drop_duplicates(subset=["url"], keep="first")
        .head(top_k)
        .reset_index(drop=True)
    )

    return unique_results


# ============================================================
# API endpoint
# ============================================================

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    try:
        results_df = search_wikipedia(
            request.question,
            request.top_k,
        )

        results = []

        for rank, (_, row) in enumerate(results_df.iterrows(), start=1):
            results.append(
                SearchResult(
                    rank=rank,
                    title=str(row["title"]),
                    url=str(row["url"]),
                    chunk_id=(
                        row["chunk_id"]
                        if "chunk_id" in row and row["chunk_id"] is not None
                        else None
                    ),
                    chunk=str(row["text"]),
                    distance=float(row["_distance"]),
                )
            )

        # Retrieve device name ('cpu' or 'cuda')
        device_info = get_device_info()

        return SearchResponse(
            question=request.question,
            device=device_info["device"],
            results=results,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health():
    device_info = get_device_info()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "database_loaded": tbl is not None,
        "device": device_info["device"],
        "device_details": device_info["details"],
    }


@app.get("/ready")
def ready():
    if model is None or tbl is None:
        raise HTTPException(status_code=503, detail="RAG service is not ready.")
    return {"status": "ready", "device": get_device_info()["device"]}


# ============================================================
# CLI Entrypoint
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wikipedia RAG FastAPI Server")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cuda", action="store_true", help="Force model to run on CUDA (GPU)")
    group.add_argument("--cpu", action="store_true", help="Force model to run on CPU")
    
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8003")), help="Port number")

    args = parser.parse_args()

    if args.cuda:
        os.environ["TARGET_DEVICE"] = "cuda"
    elif args.cpu:
        os.environ["TARGET_DEVICE"] = "cpu"

    uvicorn.run(app, host=args.host, port=args.port)