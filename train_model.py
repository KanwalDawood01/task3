import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

class SimpleRegressionNet(nn.Module):
    def __init__(self):
        super(SimpleRegressionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def fetch_dataset():
    data = fetch_california_housing()
    X = data.data
    y = data.target.reshape(-1, 1)
    return X, y

def train():
    X, y = fetch_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    model = SimpleRegressionNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    writer = SummaryWriter(log_dir="runs/housing")
    best_val_loss = float("inf")

    for epoch in range(50):
        model.train()
        total_train_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dl)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                preds = model(xb)
                loss = criterion(preds, yb)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_dl)

        writer.add_scalars("Loss", {
            "Train": avg_train_loss,
            "Val": avg_val_loss
        }, epoch)

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss
            }, "checkpoints/best_model.pt")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train()
