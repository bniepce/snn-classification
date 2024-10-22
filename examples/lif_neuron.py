from bindsnet.network import Network
from bindsnet.network.nodes import LIFNodes, Input
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
import torch

import torch
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.models import DiehlAndCook2015

time = 250

# Create the network.
network = Network()

# Create and add input, output layers.
source_layer = Input(n=1)
target_layer = LIFNodes(n=1)

network.add_layer(layer=source_layer, name="A")
network.add_layer(layer=target_layer, name="B")

forward_connection = Connection(
    source=source_layer,
    target=target_layer,
    w=0.05 + 0.1 * torch.randn(source_layer.n, target_layer.n),
)

network.add_connection(connection=forward_connection, source="A", target="B")


source_monitor = Monitor(
    obj=source_layer,
    state_vars=("s",),  # Record spikes and voltages.
    time=time,  # Length of simulation (if known ahead of time).
)
target_monitor = Monitor(
    obj=target_layer,
    state_vars=("s", "v"),  # Record spikes and voltages.
    time=time,  # Length of simulation (if known ahead of time).
)

network.add_monitor(monitor=source_monitor, name="A")
network.add_monitor(monitor=target_monitor, name="B")


input_data = torch.bernoulli(0.1 * torch.ones(time, source_layer.n)).byte()
inputs = {"A": input_data}

# Simulate network on input data.
network.run(inputs=inputs, time=time)

# Retrieve and plot simulation spike, voltage data from monitors.
spikes = {"A": source_monitor.get("s"), "B": target_monitor.get("s")}
voltages = {"B": target_monitor.get("v")}

plt.ioff()
plot_spikes(spikes)
plot_voltages(voltages, plot_type="line")
plt.show()
