import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.train.glassy_dataset import GlassyDataset


class Trainer:
    def __init__(self, batch_size: int = 128):
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()

    def train_iteration(self, model: nn.Module, optimizer: Adam, train_data: GlassyDataset,
                        device: torch.device) -> float:
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        test_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, drop_last=False)
        model.eval()
        loss = 0.
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss += self.criterion(outputs, targets)

        return loss / len(train_data)
