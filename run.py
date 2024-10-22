from argparse import ArgumentParser
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from torchvision import transforms
from src.trainer.stdp import STDPTrainer
from src.network.dc_modified import DCModified
from src.utils.parameters import get_parameter_file
import torch

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--parameter_file", type=str, required=True)
    args = parser.parse_args()
    params = get_parameter_file(args.parameter_file)

    print("\n Loading MNIST Dataset ...")

    time = params["network"]["time"]
    dt = params["network"]["dt"]

    train_data = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        root="./data",
        download=True,
        train=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    val_data = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        root="./data",
        download=True,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True)

    print("\n MNIST Dataset fully loaded ...")

    network = DCModified()
    trainer = STDPTrainer(
        network, params["training"]["epochs"], params["training"]["n_classes"]
    )
    trainer.fit(train_dataloader)
