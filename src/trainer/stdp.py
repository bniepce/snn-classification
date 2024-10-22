import torch
from tqdm import tqdm
from bindsnet.evaluation import assign_labels


class STDPTrainer(object):
    def __init__(self, network, epochs: int = 10, n_classes: int = 10):
        self.network = network
        self.epochs = epochs
        self.n_classes = n_classes

    def __run_training(self, train_loader):
        """
        Run training and evaluation of the network for one epoch.
        """
        self.network.train()
        with tqdm(total=len(train_loader), desc="Training SNN :") as pbar:
            for data in train_loader:
                image = data["encoded_image"]
                if torch.cuda.is_available():
                    inputs = {"X": image.cuda().view(self.network.time, 1, 1, 28, 28)}
                else:
                    inputs = {"X": image.view(self.network.time, 1, 1, 28, 28)}
                self.network.run(inputs, self.network.time)
                self.network.reset_state_variables()
                pbar.update()

    def fit(self, train_loader):
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            self.__run_training(train_loader)

    def predict(self, image):
        if torch.cuda.is_available():
            inputs = {"X": image.cuda().view(self.network.time, 1, 1, 28, 28)}
        else:
            inputs = {"X": image.view(self.network.time, 1, 1, 28, 28)}

        # Run network on input data
        self.network.run(inputs, time=self.network.time)
        spike_counts_per_class = torch.zeros(self.n_classes)
        for i in range(self.n_classes):
            spike_counts_per_class[i] = exc_spikes[self.neuron_labels == i].mean()

        # Predicted class is the one with the highest average spike count
        predicted_label = spike_counts_per_class.argmax().item()
