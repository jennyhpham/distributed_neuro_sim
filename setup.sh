#!/bin/bash
# Setup script for neuro_sim project

set -e

echo "Setting up neuro_sim..."

# Initialize Git LFS and pull tracked files (e.g. *.pt model files)
if command -v git-lfs &>/dev/null || git lfs version &>/dev/null 2>&1; then
    echo "Initializing Git LFS..."
    git lfs install
    git lfs pull
else
    echo "WARNING: git-lfs not found. Model files (*.pt) may be LFS pointers."
    echo "Install it with: sudo apt-get install git-lfs  (or: brew install git-lfs)"
fi

# Install package in development mode
echo "Installing package and dependencies..."
pip install -e . --break-system-packages

echo "Setup complete!"
echo ""
echo "You can now use:"
echo "  python -m neuro_sim.cli train --config configs/default.yaml"
echo "  python -m neuro_sim.cli infer"
echo ""
echo "Or after installation:"
echo "  neuro-sim-train --n-epochs 5"
echo "  neuro-sim-infer"

