import torch
import torch.nn as nn
from torch import relu
import numpy as np
import random
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

class Network(nn.Module):

    def __init__(self, in_size, layer_dims, seed=50):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


        self.layer_dims = layer_dims
        self.in_size = in_size

        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_size, layer_dims[0]))
        for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
            self.linears.append(nn.Linear(input_dim, output_dim))

        self.neurons = {}
        # a dictionary containing each layer's worth of neurons post-activation
        for index in range(1, len(layer_dims) + 1):
            # look at each hidden layer
            self.neurons[f'Hidden Layer {index} Neurons:'] = None

    def forward(self, x):
        # x - input tensor
        x = x.type(torch.FloatTensor)
        # convert into a float tensor
        x = x.reshape(-1, self.in_size)
        # turn the tensor into one tensor containing a bunch of inner tensors, each of dimension self.in_size

        for index, linear in enumerate(self.linears):
            # go through the network
            if index == len(self.linears) - 1:
                x = linear(x)
            # otherwise we use relu
            else:
                # not on the last activation
                # relu activation
                x = relu(linear(x))
            # now to keep track of the neurons
            self.neurons[f'Hidden Layer {index + 1} Neurons:'] = x

    def train_model_binary(self, train_dataset, valid_dataset, num_epochs=50, batch_size=32, learning_rate=0.001,
                    criterion=None):
        """
        Train and validate the model.

        Args:
            model: An instance of the Network class.
            train_dataset: Training dataset as a PyTorch dataset.
            valid_dataset: Validation dataset as a PyTorch dataset.
            num_epochs: Number of epochs for training.
            batch_size: Batch size for DataLoader.
            learning_rate: Learning rate for the optimizer.
            criterion: Loss function.

        Returns:
            model: The trained model.
            train_losses: List of training losses per epoch.
            valid_losses: List of validation losses per epoch.
        """
        # Default criterion
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # Track losses
        train_losses = []
        valid_losses = []

        for epoch in range(num_epochs):
            # Training phase
            self.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)

            epoch_train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)

            # Validation phase
            self.eval()
            running_valid_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in valid_loader:
                    outputs = self(batch_x)
                    loss = criterion(outputs, batch_y)
                    running_valid_loss += loss.item() * batch_x.size(0)

            epoch_valid_loss = running_valid_loss / len(valid_loader.dataset)
            valid_losses.append(epoch_valid_loss)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_valid_loss:.4f}")

        return self, train_losses, valid_losses

    def train_model_regression(self, train_dataset, valid_dataset, num_epochs=50, batch_size=32, learning_rate=0.001,
                    criterion=None):
        """
        Train and validate the model.

        Args:
            model: An instance of the Network class.
            train_dataset: Training dataset as a PyTorch dataset.
            valid_dataset: Validation dataset as a PyTorch dataset.
            num_epochs: Number of epochs for training.
            batch_size: Batch size for DataLoader.
            learning_rate: Learning rate for the optimizer.
            criterion: Loss function.

        Returns:
            model: The trained model.
            train_losses: List of training losses per epoch.
            valid_losses: List of validation losses per epoch.
        """
        # Default criterion
        if criterion is None:
            criterion = nn.MSELoss()

        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # Track losses
        train_losses = []
        valid_losses = []

        for epoch in range(num_epochs):
            # Training phase
            self.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)

            epoch_train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)

            # Validation phase
            self.eval()
            running_valid_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in valid_loader:
                    outputs = self(batch_x)
                    loss = criterion(outputs, batch_y)
                    running_valid_loss += loss.item() * batch_x.size(0)

            epoch_valid_loss = running_valid_loss / len(valid_loader.dataset)
            valid_losses.append(epoch_valid_loss)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_valid_loss:.4f}")

        return self, train_losses, valid_losses
