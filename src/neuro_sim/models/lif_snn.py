"""Leaky Integrate-and-Fire (LIF) Spiking Neural Network model using Brian2."""

from brian2 import (
    NeuronGroup,
    Synapses,
    SpikeMonitor,
    StateMonitor,
    Network,
    ms,
    mV,
    nA,
    Hz,
    defaultclock,
)


class LIFSNN:
    """A simple LIF SNN model with Brian2."""

    def __init__(
        self,
        n_neurons=100,
        tau=20 * ms,
        threshold=-50 * mV,
        reset=-65 * mV,
        v_rest=-65 * mV,
        R=50 * mV / nA,
        dt=0.1 * ms,
    ):
        """
        Initialize the LIF SNN model.

        Parameters
        ----------
        n_neurons : int
            Number of neurons in the network
        tau : Quantity
            Membrane time constant
        threshold : Quantity
            Spike threshold voltage
        reset : Quantity
            Reset voltage after spike
        v_rest : Quantity
            Resting membrane potential
        R : Quantity
            Membrane resistance
        dt : Quantity
            Simulation time step
        """
        self.n_neurons = n_neurons
        self.tau = tau
        self.threshold = threshold
        self.reset = reset
        self.v_rest = v_rest
        self.R = R
        self.dt = dt

        # Set default clock
        defaultclock.dt = dt

        # LIF neuron equations
        eqs = """
        dv/dt = (v_rest - v + R*I) / tau : volt
        I : amp
        """

        # Create neuron group with namespace for parameters
        self.neurons = NeuronGroup(
            n_neurons,
            eqs,
            threshold="v > threshold",
            reset="v = reset",
            method="euler",
            namespace={
                "tau": tau,
                "v_rest": v_rest,
                "R": R,
                "threshold": threshold,
                "reset": reset,
            },
        )
        self.neurons.v = v_rest

        # Create monitors
        self.spike_monitor = SpikeMonitor(self.neurons)
        self.state_monitor = StateMonitor(self.neurons, "v", record=True)

        # Create network
        self.net = Network(
            self.neurons,
            self.spike_monitor,
            self.state_monitor,
        )

    def add_current_input(self, neuron_indices, current):
        """
        Add constant current input to specific neurons.

        Parameters
        ----------
        neuron_indices : array-like
            Indices of neurons to stimulate
        current : Quantity
            Current amplitude
        """
        self.neurons.I[neuron_indices] = current

    def add_poisson_input(self, neuron_indices, rate):
        """
        Add Poisson spike input to specific neurons.

        Parameters
        ----------
        neuron_indices : array-like
            Indices of neurons to stimulate
        rate : Quantity
            Poisson spike rate
        """
        from brian2 import PoissonGroup, Synapses
        import numpy as np

        neuron_indices = np.asarray(neuron_indices)
        poisson = PoissonGroup(len(neuron_indices), rates=rate)
        synapses = Synapses(poisson, self.neurons, on_pre="I += 1*nA")
        # Connect each Poisson neuron to corresponding target neuron
        synapses.connect(i=range(len(neuron_indices)), j=neuron_indices)
        self.net.add(poisson, synapses)

    def add_synapses(self, source_indices, target_indices, weight=1.0 * mV, delay=1.0 * ms):
        """
        Add synaptic connections between neurons.

        Parameters
        ----------
        source_indices : array-like
            Source neuron indices
        target_indices : array-like
            Target neuron indices
        weight : Quantity
            Synaptic weight
        delay : Quantity
            Synaptic delay
        """
        import numpy as np
        
        source_indices = np.asarray(source_indices)
        target_indices = np.asarray(target_indices)
        
        # Create synapses on the full neuron group with weight parameter
        synapses = Synapses(
            self.neurons,
            self.neurons,
            model="w : volt",
            on_pre="v += w",
            delay=delay,
        )
        # Connect specific source-target pairs
        synapses.connect(i=source_indices, j=target_indices)
        # Set the synaptic weights
        synapses.w = weight
        self.net.add(synapses)

    def run(self, duration, report=None):
        """
        Run the simulation.

        Parameters
        ----------
        duration : Quantity
            Simulation duration
        report : str, optional
            Report type ('text' or None)
        """
        self.net.run(duration, report=report)

    def get_spikes(self):
        """Get spike times and neuron indices."""
        return self.spike_monitor.t, self.spike_monitor.i

    def get_voltages(self):
        """Get voltage traces."""
        return self.state_monitor.t, self.state_monitor.v

    def get_firing_rates(self, duration):
        """
        Calculate average firing rates.

        Parameters
        ----------
        duration : Quantity
            Simulation duration

        Returns
        -------
        array
            Firing rates in Hz for each neuron
        """
        spike_counts = self.spike_monitor.count
        return spike_counts / duration * Hz

