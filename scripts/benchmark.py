"""Thin shim — delegates to the neuro_sim.cli:bench console entry point.

Kept for backwards compatibility so existing shell scripts and docs that
reference `python scripts/benchmark.py` continue to work unchanged.

Preferred usage:
    neuro-sim-bench [options]            # after pip install -e .
    neuro-sim-bench --config configs/benchmark.yaml
    python scripts/benchmark.py [options]  # equivalent shim
"""

from neuro_sim.cli import bench

if __name__ == "__main__":
    bench()
