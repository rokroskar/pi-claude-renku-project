#!/usr/bin/env python3
"""Streamlit dashboard for MNIST inference and visualization."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import CNN, load_mnist

DATA_DIR = Path(os.environ.get("MNIST_DATA_DIR", "/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/home/renku/work/model-artifacts/mnist-cnn-claude"))
MODEL_PATH = MODEL_DIR / "mnist_cnn.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_FALLBACK_PATHS = [
    Path("/home/renku/work/model-artifacts/mnist-models/mnist_cnn.pt"),
    Path("/home/renku/work/pretrained-model-artifacts/mnist_cnn.pt"),
    Path("/home/renku/work/pretrained-model-artifacts/mnist-models/mnist_cnn.pt"),
]

st.set_page_config(page_title="MNIST Inference", layout="wide")
st.title("MNIST Digit Classifier")


def _find_model_path():
    for p in [MODEL_PATH] + _FALLBACK_PATHS:
        if p.exists():
            return p
    return None


def _to_tensor(imgs):
    t = torch.tensor(np.array(imgs), dtype=torch.float32)
    if t.ndim == 3:
        t = t.unsqueeze(1)
    if t.max() > 1.5:
        t = t / 255.0
    return t


@st.cache_resource
def load_model():
    path = _find_model_path()
    if path is None:
        return None
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = CNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint.get("accuracy", "?"), checkpoint.get("epoch", "?"), str(path)


@st.cache_data
def load_test_data():
    _, _, x_test, y_test = load_mnist(DATA_DIR)
    # Convert to float32 [0,1] once and keep compact
    x = np.array(x_test, dtype=np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return x, np.array(y_test, dtype=np.int64)


@st.cache_data
def compute_per_digit_accuracy(_model_key):
    """Cache keyed on model path so it recomputes only when model changes."""
    result = load_model()
    if result is None:
        return None
    model, _, _, _ = result
    x_test, y_test = load_test_data()
    t = torch.tensor(x_test).unsqueeze(1) if x_test.ndim == 3 else torch.tensor(x_test)
    ds = TensorDataset(t, torch.tensor(y_test))
    loader = DataLoader(ds, batch_size=512, num_workers=0)
    all_preds, all_true = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            all_preds.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            all_true.append(yb.numpy())
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    return {d: (all_preds[all_true == d] == d).mean() for d in range(10)}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model")
    result = load_model()
    if result:
        _, acc, epoch, model_path = result
        st.success(f"Loaded (acc={acc:.4f}, epoch {epoch})")
        st.caption(f"From: `{model_path}`")
    else:
        st.warning("No trained model found.")

    st.divider()
    st.header("Re-train")
    target_acc = st.slider("Target accuracy", 0.95, 0.999, 0.99, 0.001, format="%.3f")
    max_epochs = st.number_input("Max epochs", 5, 100, 30)
    if st.button("Start training", type="primary"):
        env = os.environ.copy()
        env["TARGET_ACCURACY"] = str(target_acc)
        env["MAX_EPOCHS"] = str(max_epochs)
        env["MNIST_DATA_DIR"] = str(DATA_DIR)
        env["MODEL_DIR"] = str(MODEL_DIR)
        with st.spinner("Training… (this may take several minutes)"):
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).parent / "train.py")],
                env=env, capture_output=True, text=True,
            )
        if proc.returncode == 0:
            st.success("Training complete!")
            load_model.clear()
            compute_per_digit_accuracy.clear()
            st.rerun()
        else:
            st.error("Training failed.")
            st.code(proc.stdout[-3000:] + "\n" + proc.stderr[-1000:])

# ── Main area ─────────────────────────────────────────────────────────────────
model_result = load_model()

try:
    x_test, y_test = load_test_data()
    data_ok = True
except Exception as e:
    st.error(f"Could not load test data: {e}")
    data_ok = False

if data_ok and model_result:
    import matplotlib.pyplot as plt

    model, acc, epoch, model_path = model_result

    st.subheader(f"Random sample predictions  —  model accuracy {acc:.4f}")

    col1, col2 = st.columns([3, 1])
    with col1:
        n_cols = st.slider("Columns", 4, 10, 6)
        n_rows = st.slider("Rows", 2, 6, 3)
    with col2:
        st.write("")
        shuffle = st.button("Shuffle samples")

    n = n_cols * n_rows
    # Regenerate indices whenever n changes or shuffle is pressed
    if shuffle or "indices" not in st.session_state or len(st.session_state["indices"]) != n:
        st.session_state["indices"] = np.random.choice(len(x_test), n, replace=False)

    idxs = st.session_state["indices"]
    imgs = x_test[idxs]       # already float32 [0,1]
    labels = y_test[idxs]

    with torch.no_grad():
        t = torch.tensor(imgs).unsqueeze(1) if imgs.ndim == 3 else torch.tensor(imgs)
        logits = model(t.to(DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 2.0))
    # axes may be 1-D if n_rows==1; always flatten
    for ax, img, label, pred, prob in zip(
        np.array(axes).flat, imgs, labels, preds, probs
    ):
        ax.imshow(img.squeeze(), cmap="gray")
        color = "green" if pred == label else "red"
        ax.set_title(f"pred={pred}\ntrue={label}\n{prob[pred]*100:.1f}%", fontsize=7, color=color)
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Per-digit accuracy (cached) ───────────────────────────────────────────
    st.subheader("Per-digit accuracy on full test set")
    digit_accs = compute_per_digit_accuracy(model_path)
    if digit_accs:
        cols = st.columns(10)
        for d, col in enumerate(cols):
            col.metric(str(d), f"{digit_accs[d]*100:.1f}%")

elif data_ok and not model_result:
    st.info("No model loaded. Use the sidebar to train one.")
