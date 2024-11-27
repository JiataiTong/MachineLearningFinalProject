import torch
import numpy as np
from torch.utils.data import DataLoader


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, seed=50):
        super(LogisticRegression, self).__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        x = x.type(torch.FloatTensor)
        x = x.reshape(-1, self.linear.in_features)
        logits = self.linear(x)
        return logits

    def train_model(
        self,
        train_dataset,
        valid_dataset,
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001,
        criterion=None,
    ):
        if criterion is None:
            criterion = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        train_losses, valid_losses = [], []

        for epoch in range(num_epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x_batch.size(0)

            train_losses.append(train_loss / len(train_loader.dataset))

            # Validation phase
            self.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in valid_loader:
                    outputs = self(x_batch)
                    loss = criterion(outputs, y_batch)
                    valid_loss += loss.item() * x_batch.size(0)

            valid_losses.append(valid_loss / len(valid_loader.dataset))

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_losses[-1]:.4f}, Validation Loss: {valid_losses[-1]:.4f}"
                )

        return self, train_losses, valid_losses
