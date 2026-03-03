FROM python:3.11-slim

# Give pip 5 minutes per download — needed for large packages like PyTorch
ENV PIP_DEFAULT_TIMEOUT=300

# git is required to install BindsNET directly from GitHub
# libgl1 and libglib2.0-0 are required by matplotlib
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch — lighter image, works in K3d without GPU setup
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (numpy, bindsnet, tqdm, pyyaml, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the project package
WORKDIR /app
COPY pyproject.toml .
COPY src/ ./src/
COPY configs/ ./configs/
RUN pip install --no-cache-dir -e .

# Bake the trained model into the image
COPY models/increasing_inhibition_network_400.pt ./models/

# Runtime directories for logs and MNIST test data
RUN mkdir -p /data/logs /data/mnist

EXPOSE 8000

# Run the FastAPI inference server
# Override NEURO_SIM_CONFIG / NEURO_SIM_MODEL env vars to swap model at deploy time
CMD ["uvicorn", "neuro_sim.api.server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
