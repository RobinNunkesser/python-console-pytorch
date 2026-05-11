import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs_head: int = 2
    epochs_finetune: int = 2
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    data_dir: str = "./data"


def build_dataloaders(config: TrainingConfig):
    # ImageNet pretraining expects 224x224 and ImageNet normalization.
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.CIFAR10(root=config.data_dir, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root=config.data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def replace_classifier(model: nn.Module, num_classes: int = 10) -> nn.Module:
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_requires_grad(model: nn.Module, enabled: bool) -> None:
    for param in model.parameters():
        param.requires_grad = enabled


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


def run_transfer_learning(config: TrainingConfig):
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(config)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = replace_classifier(model, num_classes=10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Phase 1: train only the classification head.
    set_requires_grad(model, False)
    set_requires_grad(model.fc, True)
    optimizer = optim.Adam(model.fc.parameters(), lr=config.lr_head)

    print("\nPhase 1: Train classifier head")
    for epoch in range(1, config.epochs_head + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch}/{config.epochs_head} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

    # Phase 2: fine-tune last residual block + head.
    set_requires_grad(model, False)
    set_requires_grad(model.layer4, True)
    set_requires_grad(model.fc, True)
    optimizer = optim.Adam(
        list(model.layer4.parameters()) + list(model.fc.parameters()),
        lr=config.lr_finetune,
    )

    print("\nPhase 2: Fine-tune layer4 + head")
    for epoch in range(1, config.epochs_finetune + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch}/{config.epochs_finetune} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Learning with ResNet18 on CIFAR-10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-head", type=int, default=2)
    parser.add_argument("--epochs-finetune", type=int, default=2)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    cfg = TrainingConfig(
        batch_size=args.batch_size,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        lr_head=args.lr_head,
        lr_finetune=args.lr_finetune,
        data_dir=args.data_dir,
    )
    run_transfer_learning(cfg)
