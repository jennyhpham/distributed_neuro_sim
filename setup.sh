#!/bin/bash
# Setup script for neuro_sim project

set -e

echo "Setting up neuro_sim..."

# Install package in development mode
echo "Installing package and dependencies..."
pip install -e .

echo "Setup complete!"
echo ""
echo "You can now use:"
echo "  python -m neuro_sim.cli train --config configs/default.yaml"
echo "  python -m neuro_sim.cli infer"
echo ""
echo "Or after installation:"
echo "  neuro-sim-train --n-epochs 5"
echo "  neuro-sim-infer"

