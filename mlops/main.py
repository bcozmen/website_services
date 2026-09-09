import argparse
import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import numpy as np
import torch
import threading
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import HTTPException, Query

# Import your database components
from database import SessionLocal, ClickedPoint
# Import your models (assuming they are in models.py or defined above)
from models import KDEModel, NNModel  

# Lock to prevent concurrent training issues with global models
model_lock = threading.Lock()

# Models are created during startup, after the requested device has been
# selected. This keeps importing the module side-effect free.
kde_model = None
nn_model = None
device = "cpu"
models_initialized = False


def get_device_info() -> dict:
    """Return the execution device used by the neural density model."""
    if not models_initialized:
        return {"device": "uninitialized", "details": None}

    details = (
        torch.cuda.get_device_name(torch.device(device))
        if device == "cuda"
        else "CPU"
    )
    return {"device": device, "details": details}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global kde_model, nn_model, device, models_initialized

    requested_device = os.getenv("TARGET_DEVICE", "").lower()
    if requested_device not in {"", "cpu", "cuda"}:
        raise RuntimeError("TARGET_DEVICE must be either 'cpu' or 'cuda'.")
    if requested_device:
        device = requested_device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    print(f"Loading MLOps models on '{device}'...")
    kde_model = KDEModel(bandwidth=0.04)
    model_path = Path(
        os.getenv(
            "NN_MODEL_PATH",
            str(Path(__file__).resolve().parent / "nn_density_model.pth"),
        )
    )
    nn_model = NNModel(
        bandwidth=0.03,
        grid_res=40,
        device=device,
        model_path=model_path,
    )

    # Load old datapoints from the database on startup
    db = SessionLocal()
    try:
        points = db.query(ClickedPoint).all()
        if len(points) >= 2:
            coords_np = np.array([[p.x, p.y] for p in points], dtype=np.float32)
            coords_torch = torch.tensor(coords_np, dtype=torch.float32, device=device)
            with model_lock:
                kde_model(coords_np)
                nn_model.train(coords_torch, epochs=20)
    finally:
        db.close()

    models_initialized = True
    print(f"MLOps models loaded on '{device}'.")
    yield

    kde_model = None
    nn_model = None
    models_initialized = False

app = FastAPI(
    title="Spatial Density API",
    description="Interactive KDE and neural density estimation.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS so your Django frontend can communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your Django domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas for data validation
class PointCreate(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)

class PointResponse(BaseModel):
    id: int
    x: float
    y: float

    class Config:
        from_attributes = True

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health():
    device_info = get_device_info()
    return {
        "status": "ok",
        "model_loaded": models_initialized,
        "models_loaded": models_initialized,
        "device": device_info["device"],
        "device_details": device_info["details"],
    }


@app.get("/ready")
def ready():
    if not models_initialized or kde_model is None or nn_model is None:
        raise HTTPException(status_code=503, detail="MLOps service is not ready.")
    return {"status": "ready", "device": device}


@app.post("/points/", response_model=PointResponse)
def add_point(point: PointCreate, db: Session = Depends(get_db)):
    """Registers a new clicked point coordinate."""
    db_point = ClickedPoint(x=point.x, y=point.y)
    db.add(db_point)
    db.commit()
    db.refresh(db_point)
    return db_point


@app.get("/points/", response_model=list[PointResponse])
def get_points(db: Session = Depends(get_db)):
    """Returns all registered points."""
    return db.query(ClickedPoint).all()


@app.post("/train-and-evaluate/")
def train_and_evaluate(
    grid_size: int = Query(default=100, ge=10, le=300),
    db: Session = Depends(get_db),
):
    if not models_initialized or nn_model is None or kde_model is None:
        raise HTTPException(status_code=503, detail="MLOps models are not initialized.")

    points = db.query(ClickedPoint).all()
    
    if len(points) < 2:
        return {"status": "insufficient_data", "kde": None, "nn": None}

    coords_np = np.array([[p.x, p.y] for p in points], dtype=np.float32)
    coords_torch = torch.tensor(coords_np, dtype=torch.float32, device=device)

    with model_lock:
        # 1. Process KDE
        kde_model(coords_np)
        _, _, kde_Z = kde_model.visualize(grid_size=grid_size)
    
        # 2. Process NN
        nn_model.train(coords_torch, epochs=20) 
        _, _, nn_Z = nn_model.visualize(grid_size=grid_size)

    # Send clean 1D axis arrays along with the 2D Z density matrix
    return {
        "status": "success",
        "total_points": len(points),
        "axis_range": np.linspace(0, 1, grid_size).tolist(),  # Clean [0, 1] 1D array
        "kde_Z": kde_Z.tolist(),
        "nn_Z": nn_Z.tolist()
    }


@app.delete("/points/clear/")
def clear_points(db: Session = Depends(get_db)):
    """Clears the database to restart the experiment."""
    db.query(ClickedPoint).delete()
    db.commit()
    return {"message": "All points cleared successfully."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Spatial Density FastAPI Server")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cuda", action="store_true", help="Force model to run on CUDA (GPU)")
    group.add_argument("--cpu", action="store_true", help="Force model to run on CPU")

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8002")), help="Port number")
    args = parser.parse_args()

    if args.cuda:
        os.environ["TARGET_DEVICE"] = "cuda"
    elif args.cpu:
        os.environ["TARGET_DEVICE"] = "cpu"

    uvicorn.run(app, host=args.host, port=args.port)