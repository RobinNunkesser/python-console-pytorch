import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Config:
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3


class AdultDataset(Dataset):
    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


class TabularEmbeddingModel(nn.Module):
    def __init__(self, num_numeric: int, cat_cardinalities: list[int], hidden_dim: int = 128):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, min(50, (cardinality + 1) // 2))
            for cardinality in cat_cardinalities
        ])

        emb_dim_total = sum(emb.embedding_dim for emb in self.embeddings)
        input_dim = num_numeric + emb_dim_total

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + cat_embs, dim=1)
        logits = self.mlp(x).squeeze(1)
        return logits


def load_and_prepare_data():
    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame.copy()

    # Target: >50K => 1, <=50K => 0
    y = (df["class"] == ">50K").astype(np.float32).values
    X = df.drop(columns=["class"])

    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Fill missing and encode categoricals per column.
    X_cat = X[cat_cols].astype("string").fillna("Unknown").copy()
    cat_cardinalities = []
    cat_arrays = []

    for col in cat_cols:
        codes, uniques = pd.factorize(X_cat[col], sort=True)
        cat_arrays.append(codes)
        cat_cardinalities.append(len(uniques) + 1)

    x_cat = np.vstack(cat_arrays).T.astype(np.int64)

    # Scale numeric columns.
    X_num = X[num_cols].copy()
    for col in num_cols:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
    X_num = X_num.fillna(X_num.median(numeric_only=True))

    scaler = StandardScaler()
    x_num = scaler.fit_transform(X_num.values).astype(np.float32)

    x_num_train, x_num_test, x_cat_train, x_cat_test, y_train, y_test = train_test_split(
        x_num, x_cat, y, test_size=0.2, random_state=SEED, stratify=y
    )

    x_num_train, x_num_val, x_cat_train, x_cat_val, y_train, y_val = train_test_split(
        x_num_train, x_cat_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    return (
        x_num_train,
        x_cat_train,
        y_train,
        x_num_val,
        x_cat_val,
        y_val,
        x_num_test,
        x_cat_test,
        y_test,
        len(num_cols),
        cat_cardinalities,
    )


def run(config: Config):
    device = get_device()
    print(f"Using device: {device}")

    (
        x_num_train,
        x_cat_train,
        y_train,
        x_num_val,
        x_cat_val,
        y_val,
        x_num_test,
        x_cat_test,
        y_test,
        num_numeric,
        cat_cardinalities,
    ) = load_and_prepare_data()

    train_ds = AdultDataset(x_num_train, x_cat_train, y_train)
    val_ds = AdultDataset(x_num_val, x_cat_val, y_val)
    test_ds = AdultDataset(x_num_test, x_cat_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    model = TabularEmbeddingModel(num_numeric=num_numeric, cat_cardinalities=cat_cardinalities).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_num_b, x_cat_b, y_b in train_loader:
            x_num_b = x_num_b.to(device)
            x_cat_b = x_cat_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits = model(x_num_b, x_cat_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y_b.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y_b).sum().item()
            train_total += y_b.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x_num_b, x_cat_b, y_b in val_loader:
                x_num_b = x_num_b.to(device)
                x_cat_b = x_cat_b.to(device)
                y_b = y_b.to(device)

                logits = model(x_num_b, x_cat_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * y_b.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y_b).sum().item()
                val_total += y_b.size(0)

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss / train_total:.4f}, train_acc={train_correct / train_total:.4f}, "
            f"val_loss={val_loss / val_total:.4f}, val_acc={val_correct / val_total:.4f}"
        )

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x_num_b, x_cat_b, y_b in test_loader:
            x_num_b = x_num_b.to(device)
            x_cat_b = x_cat_b.to(device)
            y_b = y_b.to(device)

            logits = model(x_num_b, x_cat_b)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            test_correct += (preds == y_b).sum().item()
            test_total += y_b.size(0)

    print(f"Test accuracy: {test_correct / test_total:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular DL with embeddings on Adult dataset")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = Config(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
    run(cfg)
