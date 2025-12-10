"""Simple LIF SNN example with Brian2."""

import matplotlib.pyplot as plt
from brian2 import ms, mV, nA, Hz

from neuro_sim.models.lif_snn import LIFSNN


def main():
    """Run a simple LIF SNN simulation."""
    # Create LIF network
    snn = LIFSNN(
        n_neurons=100,
        tau=20 * ms,
        threshold=-50 * mV,
        reset=-65 * mV,
        v_rest=-65 * mV,
        R=50 * mV / nA,
        dt=0.1 * ms,
    )

    # Add constant current input to first 10 neurons
    snn.add_current_input(range(10), 0.5 * nA)

    # Add Poisson input to neurons 10-20
    snn.add_poisson_input(range(10, 20), 50 * Hz)

    # Add random synaptic connections
    import numpy as np

    np.random.seed(42)
    n_connections = 50
    source_indices = np.random.randint(0, 100, n_connections)
    target_indices = np.random.randint(0, 100, n_connections)
    # Avoid self-connections
    mask = source_indices != target_indices
    snn.add_synapses(
        source_indices[mask],
        target_indices[mask],
        weight=2.0 * mV,
        delay=1.0 * ms,
    )

    # Run simulation
    duration = 200 * ms
    print(f"Running simulation for {duration}...")
    snn.run(duration, report="text")

    # Get results
    spike_times, spike_indices = snn.get_spikes()
    voltage_times, voltages = snn.get_voltages()
    firing_rates = snn.get_firing_rates(duration)

    # Print statistics
    print(f"\nSimulation complete!")
    print(f"Total spikes: {len(spike_times)}")
    print(f"Average firing rate: {firing_rates.mean():.2f} Hz")
    print(f"Max firing rate: {firing_rates.max():.2f} Hz")
    print(f"Min firing rate: {firing_rates.min():.2f} Hz")

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Raster plot
    axes[0].scatter(spike_times / ms, spike_indices, s=1, alpha=0.6)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Neuron index")
    axes[0].set_title("Spike Raster Plot")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Voltage traces for first 5 neurons
    for i in range(min(5, snn.n_neurons)):
        axes[1].plot(voltage_times / ms, voltages[i] / mV, label=f"Neuron {i}", alpha=0.7)
    axes[1].axhline(y=snn.threshold / mV, color="r", linestyle="--", label="Threshold")
    axes[1].axhline(y=snn.reset / mV, color="g", linestyle="--", label="Reset")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Voltage (mV)")
    axes[1].set_title("Voltage Traces (First 5 Neurons)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lif_snn_results.png", dpi=150)
    print("\nResults saved to 'lif_snn_results.png'")
    plt.show()


if __name__ == "__main__":
    main()

