"""FastAPI inference server for the SNN neuromorphic workload.

Exposes a single-image MNIST digit prediction endpoint backed by a
Spiking Neural Network (BindsNET/Brian2). Designed as the measurement
interface for container orchestration benchmarking.

Endpoints:
  POST /predict   — upload a grayscale image, get predicted digit + SNN metrics
  GET  /health    — liveness/readiness probe for Kubernetes
  GET  /metrics   — Prometheus scrape endpoint (auto-instrumented)
"""

import asyncio
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity
import bindsnet.evaluation as bindsnet_eval
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator

from neuro_sim.config import Config
from neuro_sim.training.evaluator import Evaluator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (loaded once at startup)
# ---------------------------------------------------------------------------

_evaluator: Optional[Evaluator] = None
_encoder: Optional[PoissonEncoder] = None
_lock = asyncio.Lock()

_DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "som_lm_snn.yaml"
_DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "increasing_inhibition_network_400.pt"


def _load_evaluator(config_path: str, model_path: str) -> Evaluator:
    config = Config(config_path=config_path)
    # Force batch_size=1 and CPU for the API (no GPU needed for single-image inference)
    config.set("inference.batch_size", 1)
    config.set("inference.gpu", False)
    config.set("training.gpu", False)
    return Evaluator(config, model_path=model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release on shutdown."""
    global _evaluator, _encoder

    cfg_path = os.environ.get("NEURO_SIM_CONFIG", str(_DEFAULT_CONFIG))
    mdl_path = os.environ.get("NEURO_SIM_MODEL", str(_DEFAULT_MODEL))

    logger.info(f"Loading model from: {mdl_path}")
    logger.info(f"Using config: {cfg_path}")

    _evaluator = _load_evaluator(cfg_path, mdl_path)

    time_window = _evaluator.config.inference["time"]
    dt = _evaluator.config.model["dt"]
    _encoder = PoissonEncoder(time=time_window, dt=dt)

    logger.info(
        f"Server ready — model: {_evaluator.model_wrapper.model_name}, "
        f"device: {_evaluator.device}, sim_window: {time_window}ms"
    )
    yield
    logger.info("Server shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NeuroSim SNN Inference API",
    description=(
        "Spiking Neural Network inference endpoint for MNIST digit recognition. "
        "Part of a thesis on container-based orchestration of neuromorphic workloads."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness and readiness probe for Kubernetes."""
    if _evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model": _evaluator.model_wrapper.model_name,
        "n_neurons": _evaluator.config.model["n_neurons"],
        "simulation_time_ms": _evaluator.config.inference["time"],
        "device": str(_evaluator.device),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the digit in an uploaded image using the SNN.

    The image is Poisson-encoded into spike trains and run through the
    network for `simulation_time_ms` milliseconds of simulated time.

    Args:
        file: A grayscale (or RGB) image of a handwritten digit.
              It will be resized to 28×28 internally.

    Returns:
        digit                  — predicted digit (0–9) via proportion-weighting
        all_activity_digit     — predicted digit via winner-takes-all spike count
        inference_time_ms      — wall-clock latency for encoding + simulation + decoding
        simulation_time_ms     — fixed SNN time window (neuromorphic floor latency)
        spike_count            — total output-layer spikes (SNN workload intensity)
    """
    if _evaluator is None or _encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read and validate image
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}")

    # Only one inference can run at a time — the SNN network is stateful
    async with _lock:
        t_start = time.perf_counter()

        # --- Encoding ---
        intensity = _evaluator.config.inference["intensity"]
        time_window = _evaluator.config.inference["time"]

        # PIL → numpy → tensor scaled to [0, intensity], shape (1, 28, 28)
        img_arr = np.array(img, dtype=np.float32)
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0) * (intensity / 255.0)

        # PoissonEncoder: (1, 28, 28) → (time, 1, 28, 28)
        encoded = _encoder(img_tensor)

        # Add batch dimension: (time, 1, 28, 28) → (time, 1, 1, 28, 28)
        encoded = encoded.unsqueeze(1)

        # --- SNN simulation ---
        _evaluator.network.reset_state_variables()
        _evaluator.network.run(inputs={"X": encoded}, time=time_window)

        # --- Read output spikes ---
        output_layer = _evaluator.output_layer
        # Monitor output shape: (time_steps, batch=1, n_neurons)
        # After permute(1, 0, 2): (batch=1, time_steps, n_neurons)
        spike_record = _evaluator.spikes[output_layer].get("s").permute((1, 0, 2))

        if spike_record.is_sparse:
            spike_record = spike_record.to_dense()

        assignments = _evaluator.assignments
        proportions = _evaluator.proportions

        # --- Decode predictions ---
        all_pred = all_activity(
            spikes=spike_record,
            assignments=assignments,
            n_labels=10,
        )
        prop_pred = bindsnet_eval.proportion_weighting(
            spikes=spike_record,
            assignments=assignments,
            proportions=proportions,
            n_labels=10,
        )

        t_end = time.perf_counter()

    return JSONResponse({
        "digit": int(prop_pred[0].item()),
        "all_activity_digit": int(all_pred[0].item()),
        "inference_time_ms": round((t_end - t_start) * 1000, 2),
        "simulation_time_ms": time_window,
        "spike_count": int(spike_record.sum().item()),
    })
