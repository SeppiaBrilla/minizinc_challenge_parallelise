from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from copy import deepcopy
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch

class NN(BaseEstimator, ClassifierMixin):
    def __init__(self, layers: list[int], lr: float = 0.01, batch_size: int = 32, patience: int = 5):
        """
        layers: list of layer sizes
        lr: learning rate
        batch_size: mini-batch size
        patience: number of epochs to wait for improvement before stopping
        """
        self.layers = layers
        self.n_epochs = 10000
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self._build_model()

    def _build_model(self):
        nn_layers = []
        for i in range(1, len(self.layers)):
            nn_layers.append(nn.Linear(self.layers[i-1], self.layers[i]))
            if i < len(self.layers) - 1:
                nn_layers.append(nn.LeakyReLU())
        self.nn = nn.Sequential(*nn_layers)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, verbose=False):
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_np = y.values if isinstance(y, pd.DataFrame) else np.array(y)

        X_train, X_val, y_train, y_val = train_test_split(
            X_np, y_np, test_size=0.1, stratify=y_np
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.nn.parameters(), lr=self.lr)

        best_loss = float("inf")
        best_model = deepcopy(self.nn.state_dict())
        epochs_no_improve = 0

        for epoch in range(self.n_epochs):
            self.nn.train()
            tot_loss = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.nn(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                tot_loss.append(float(loss.detach()))

            self.nn.eval()
            with torch.no_grad():
                val_outputs = self.nn(X_val)
                val_loss = criterion(val_outputs, y_val).item()

            if verbose and epoch % 10 == 0:
                loss = np.mean(tot_loss)
                print(f"{epoch}/{self.n_epochs} - ({loss:.4f}, {val_loss:.4f})")
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.nn.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                if verbose:
                    print("early stopping")
                break

        self.nn.load_state_dict(best_model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.nn.eval()
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        with torch.no_grad():
            logits = self.nn(torch.tensor(X_np, dtype=torch.float32))
            probs = torch.sigmoid(logits).numpy().flatten()
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.nn.eval()
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        with torch.no_grad():
            logits = self.nn(torch.tensor(X_np, dtype=torch.float32))
            probs = torch.sigmoid(logits).numpy().flatten()
        return np.vstack([1 - probs, probs]).T

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X))
