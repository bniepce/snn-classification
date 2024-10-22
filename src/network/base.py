from bindsnet.network import Network
import torch


class CustomSNN(Network):
    def __init__(self, dt: float = 1.0):
        super().__init__(dt=dt)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def summary(self):
        """
        Print a summary of the network architecture
        """
        print("\n Network Architecture \n")
        print("-" * 65)
        print(f"{'Layer Name':<20}{'Type':<20}{'Neurons':<10}{'Shape':<15}")
        print("-" * 65)

        # Print layer info
        for layer_name, layer in self.layers.items():
            layer_type = layer.__class__.__name__
            neurons = layer.n
            shape = layer.shape if hasattr(layer, "shape") else "N/A"
            print(f"{layer_name:<20}{layer_type:<20}{neurons:<10}{str(shape):<15}")

        print("\nConnections")
        print("-" * 80)
        print(f"{'Source':<15}{'Target':<15}{'Shape':<30}{'Connection Type':<20}")
        print("-" * 80)

        # Print connection info
        for conn in self.connections:
            source = conn[0]
            target = conn[1]
            current_connection = self.connections[(source, target)]
            update_rule_name = current_connection.update_rule.__class__.__name__

            conn_type = current_connection.__class__.__name__
            shape = self.connections[(source, target)].w.shape
            print(
                f"{source:<15}{target:<15}{str(shape):<30}{update_rule_name} {conn_type:<20}"
            )

        print("\nMonitors")
        print("-" * 50)
        print(f"{'Layer Name':<20}{'Type':<20}{'Time Steps':<10}")
        print("-" * 50)

        # Print monitor info
        for monitor_name, monitor in self.monitors.items():
            monitor_type = monitor.__class__.__name__
            time_steps = monitor.time
            print(f"{monitor_name:<20}{monitor_type:<20}{time_steps:<10}")
        print("-" * 50)
        print("\n")
