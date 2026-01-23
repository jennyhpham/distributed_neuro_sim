FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python-is-python3 \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126 \
    --break-system-packages

COPY requirements.txt .
RUN pip install -r requirements.txt --break-system-packages

WORKDIR /workspaces/

# Copy the project source and install the package
# Note: During devcontainer build, files are copied, but for live editing,
# the postCreateCommand in devcontainer.json will also install the package
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install -e . --break-system-packages || true
