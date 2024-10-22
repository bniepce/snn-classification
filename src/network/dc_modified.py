import torch
from .base import CustomSNN
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import LIFNodes, Input, DiehlAndCookNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import PostPre
from typing import Optional, Union, Sequence, Iterable


class DCModified(CustomSNN):
    def __init__(
        self,
        n_input: int = 784,
        dt: float = 1.0,
        n_neurons: int = 100,
        w_exc_strength: float = 22.5,
        w_inh_strength: float = 17.5,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        input_shape: Optional[Iterable[int]] = (1, 28, 28),
        time: int = 250,
    ):
        """
        Constructor for class ``DCModified``.

        :param n_input: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param w_exc_strength: Strength of synapse weights from excitatory to inhibitory layer.
        :param w_inh_strength: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param input_shape: The dimensionality of the input layer.
        :param time: The simulation time of an input
        """
        super().__init__(dt=dt)
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.w_exc_strength = w_exc_strength
        self.w_inh_strength = w_inh_strength
        self.nu = nu
        self.wmin = wmin
        self.wmax = wmax
        self.norm = norm
        self.theta_plus = theta_plus
        self.tc_theta_decay = tc_theta_decay
        self.input_shape = input_shape
        self.time = time

        self.__init_layers()
        self.__init_synapses()
        self.__init_monitors()
        self.summary()

    def __init_layers(self):
        in_layer = Input(
            n=self.n_input, shape=self.input_shape, traces=True, tc_trace=20.0
        )
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=self.theta_plus,
            tc_theta_decay=self.tc_theta_decay,
        )
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=-40.0,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )

        self.add_layer(in_layer, name="Input")
        self.add_layer(
            exc_layer,
            name="Excitatory",
        )
        self.add_layer(inh_layer, name="Inhibitory")

    def __init_synapses(self):
        w = 0.3 * torch.rand(self.n_input, self.n_neurons)
        stdp_conn = Connection(
            source=self.layers["Input"],
            target=self.layers["Excitatory"],
            w=w,
            update_rule=PostPre,
            nu=self.nu,
            reduction=None,
            wmin=self.wmin,
            wmax=self.wmax,
            norm=self.norm,
        )
        w = self.w_exc_strength * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=self.layers["Excitatory"],
            target=self.layers["Inhibitory"],
            w=w,
            wmin=0,
            wmax=self.w_exc_strength,
        )
        w = -self.w_inh_strength * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=self.layers["Inhibitory"],
            target=self.layers["Excitatory"],
            w=w,
            wmin=-self.w_inh_strength,
            wmax=0,
        )

        self.add_connection(stdp_conn, source="Input", target="Excitatory")
        self.add_connection(exc_inh_conn, source="Excitatory", target="Inhibitory")
        self.add_connection(inh_exc_conn, source="Inhibitory", target="Excitatory")

    def __init_monitors(self):
        """
        Add monitors to hidden and output layers to catch spiking activity
        """
        exc_monitor = Monitor(
            self.layers["Excitatory"],
            state_vars=["s"],
            time=self.time,
            device=self.device,
        )
        self.add_monitor(exc_monitor, name="Exc_Monitor")
