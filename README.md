# NeuroSim

A clean Python project for simulating Spiking Neural Networks (SNNs) using Brian2.

## Features

- LIF (Leaky Integrate-and-Fire) neuron models
- Simple and extensible architecture
- Example simulations included

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the simple LIF example:

```bash
python -m neuro_sim.main
```

Or run the example directly:

```bash
python -m neuro_sim.examples.simple_lif
```

## Project Structure

```
neuro_sim/
├── src/
│   └── neuro_sim/
│       ├── __init__.py
│       ├── main.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── lif_snn.py      # LIF SNN model
│       └── examples/
│           ├── __init__.py
│           └── simple_lif.py   # Example simulation
├── requirements.txt
└── README.md
```

