#!/usr/bin/env python3
"""MNIST CNN training. Exits 0 when test accuracy >= TARGET_ACCURACY."""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model import CNN, load_mnist, make_loaders, evaluate

DATA_DIR = Path(os.environ.get("MNIST_DATA_DIR", "/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/home/renku/work/model-artifacts/mnist-cnn-claude"))
MODEL_PATH = MODEL_DIR / "mnist_cnn.pt"
TARGET_ACCURACY = float(os.environ.get("TARGET_ACCURACY", "0.99"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "30"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
LR = float(os.environ.get("LR", "0.001"))


def main():
    print(f"Data dir  : {DATA_DIR}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Target acc: {TARGET_ACCURACY}")

    x_train, y_train, x_test, y_test = load_mnist(DATA_DIR)
    print(f"Loaded: train={x_train.shape}  test={x_test.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = make_loaders(x_train, y_train, x_test, y_test, BATCH_SIZE)

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 10,
        steps_per_epoch=len(train_loader), epochs=MAX_EPOCHS,
    )
    criterion = nn.CrossEntropyLoss()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:2d}/{MAX_EPOCHS} | loss={avg_loss:.4f} | test_acc={acc:.4f}", flush=True)

        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state": model.state_dict(), "accuracy": acc, "epoch": epoch}, MODEL_PATH)
            print(f"  -> Saved model (best_acc={acc:.4f})", flush=True)

        if acc >= TARGET_ACCURACY:
            print(f"\nTarget accuracy {TARGET_ACCURACY} reached at epoch {epoch}. Done.", flush=True)
            sys.exit(0)

    print(f"\nTraining finished. Best accuracy: {best_acc:.4f}", flush=True)
    sys.exit(0 if best_acc >= TARGET_ACCURACY else 1)


if __name__ == "__main__":
    main()
