"""Shared CNN architecture and MNIST data-loading utilities."""

import gzip
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _read_idx_bytes(data: bytes, is_image: bool):
    import io
    buf = io.BytesIO(data)
    if is_image:
        magic, n, rows, cols = struct.unpack(">IIII", buf.read(16))
        assert magic == 2051, f"Bad IDX magic: {magic}"
        return np.frombuffer(buf.read(), dtype=np.uint8).reshape(n, rows, cols)
    else:
        magic, n = struct.unpack(">II", buf.read(8))
        assert magic == 2049, f"Bad IDX magic: {magic}"
        return np.frombuffer(buf.read(), dtype=np.uint8)


def _open_idx(path: Path, is_image: bool):
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return _read_idx_bytes(raw, is_image)


def _load_idx_set(data_dir: Path):
    name_map = {
        "train_images": [
            "train-images-idx3-ubyte", "train-images-idx3-ubyte.gz",
            "train-images.idx3-ubyte",
        ],
        "train_labels": [
            "train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz",
            "train-labels.idx1-ubyte",
        ],
        "test_images": [
            "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz",
            "t10k-images.idx3-ubyte",
        ],
        "test_labels": [
            "t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz",
            "t10k-labels.idx1-ubyte",
        ],
    }
    found = {}
    for key, names in name_map.items():
        for name in names:
            hits = list(data_dir.rglob(name))
            if hits:
                found[key] = hits[0]
                break
    if len(found) != 4:
        return None
    return (
        _open_idx(found["train_images"], True),
        _open_idx(found["train_labels"], False),
        _open_idx(found["test_images"], True),
        _open_idx(found["test_labels"], False),
    )


def _load_npy_set(data_dir: Path):
    """Load the Written-and-Spoken-Digits Zenodo format (data_wr_*.npy + labels_*.npy)."""
    train_img = data_dir / "data_wr_train.npy"
    train_lbl = data_dir / "labels_train.npy"
    test_img  = data_dir / "data_wr_test.npy"
    test_lbl  = data_dir / "labels_test.npy"
    if not all(p.exists() for p in [train_img, train_lbl, test_img, test_lbl]):
        return None

    def _load(path):
        arr = np.load(path, allow_pickle=True)
        if arr.dtype.kind == 'f':
            # Zenodo record stores float64 0-255; convert to uint8
            arr = arr.astype(np.float32)
            if arr.max() > 1.5:
                arr = arr / 255.0
        return arr

    x_tr = _load(train_img)
    y_tr = np.load(train_lbl, allow_pickle=True).astype(np.int64)
    x_te = _load(test_img)
    y_te = np.load(test_lbl, allow_pickle=True).astype(np.int64)
    return x_tr, y_tr, x_te, y_te


def load_mnist(data_dir: Path):
    """Load MNIST from *data_dir*, auto-detecting npy, IDX, or .npz format."""
    data_dir = Path(data_dir)

    # Zenodo 10.5281/zenodo.3515935 format: data_wr_*.npy + labels_*.npy
    result = _load_npy_set(data_dir)
    if result:
        return result

    # Standard .npz (keras-style or custom)
    for npz in data_dir.rglob("*.npz"):
        try:
            d = np.load(npz, allow_pickle=True)
            keys = list(d.keys())
            if "x_train" in keys:
                return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
            arrays = {k: d[k] for k in keys}
            imgs = sorted(
                [(k, v) for k, v in arrays.items() if v.ndim in (2, 3)],
                key=lambda x: -x[1].shape[0],
            )
            lbls = sorted(
                [(k, v) for k, v in arrays.items() if v.ndim == 1],
                key=lambda x: -x[1].shape[0],
            )
            if len(imgs) >= 2 and len(lbls) >= 2:
                return imgs[0][1], lbls[0][1], imgs[1][1], lbls[1][1]
        except Exception as e:
            print(f"[warn] Could not load {npz}: {e}")

    # Raw IDX files (plain or gzipped)
    result = _load_idx_set(data_dir)
    if result:
        return result

    files = [str(p) for p in data_dir.rglob("*") if p.is_file()][:30]
    raise FileNotFoundError(
        f"Could not find MNIST data in {data_dir}.\n"
        f"Expected data_wr_*.npy, IDX files, or mnist.npz.  Found: {files}"
    )


def make_loaders(x_train, y_train, x_test, y_test, batch_size: int):
    def _ds(x, y):
        t = torch.tensor(x, dtype=torch.float32)
        if t.ndim == 3:
            t = t.unsqueeze(1)
        if t.max() > 1.5:   # uint8 or float 0-255 → normalise
            t = t / 255.0
        return TensorDataset(t, torch.tensor(y, dtype=torch.long))

    return (
        DataLoader(_ds(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(_ds(x_test, y_test), batch_size=batch_size, num_workers=2, pin_memory=True),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / total
