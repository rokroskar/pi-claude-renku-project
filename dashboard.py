#!/usr/bin/env python3
"""Streamlit dashboard for MNIST inference and visualization."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from model import CNN, load_mnist, make_loaders, evaluate

DATA_DIR = Path(os.environ.get("MNIST_DATA_DIR", "/home/renku/work/data/mnist"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/home/renku/work/models"))
MODEL_PATH = MODEL_DIR / "mnist_cnn.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="MNIST Inference", layout="wide")
st.title("MNIST Digit Classifier")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = CNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint.get("accuracy", "?"), checkpoint.get("epoch", "?")


@st.cache_data
def load_test_data():
    x_train, y_train, x_test, y_test = load_mnist(DATA_DIR)
    return x_test, y_test


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model")
    result = load_model()
    if result:
        _, acc, epoch = result
        st.success(f"Model loaded (acc={acc:.4f}, epoch {epoch})")
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
        with st.spinner("Training…"):
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).parent / "train.py")],
                env=env, capture_output=True, text=True,
            )
        if proc.returncode == 0:
            st.success("Training complete!")
            load_model.clear()
            st.rerun()
        else:
            st.error("Training failed.")
            st.code(proc.stdout[-3000:] + proc.stderr[-1000:])

# ── Main area ────────────────────────────────────────────────────────────────
model_result = load_model()

try:
    x_test, y_test = load_test_data()
    data_ok = True
except Exception as e:
    st.error(f"Could not load test data: {e}")
    data_ok = False

if data_ok and model_result:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    model, acc, epoch = model_result

    st.subheader(f"Random sample predictions  —  model accuracy {acc:.4f}")

    n_cols = st.slider("Columns", 4, 10, 6)
    n_rows = st.slider("Rows", 2, 6, 3)
    n = n_cols * n_rows

    if st.button("Shuffle samples") or "indices" not in st.session_state:
        st.session_state["indices"] = np.random.choice(len(x_test), n, replace=False)

    idxs = st.session_state["indices"]
    imgs = x_test[idxs]
    labels = y_test[idxs]

    # batch inference
    t = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1) / 255.0
    with torch.no_grad():
        logits = model(t.to(DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 2.0))
    for ax, img, label, pred, prob in zip(axes.flat, imgs, labels, preds, probs):
        ax.imshow(img, cmap="gray")
        color = "green" if pred == label else "red"
        ax.set_title(f"pred={pred}\ntrue={label}\n{prob[pred]*100:.1f}%", fontsize=7, color=color)
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Confusion-style per-digit accuracy ──────────────────────────────────
    st.subheader("Per-digit accuracy on full test set")
    t_all = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1) / 255.0
    _, test_loader = make_loaders(
        x_test, y_test, x_test, y_test, batch_size=256
    )
    all_preds, all_true = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            all_preds.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            all_true.append(yb.numpy())
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    digit_accs = {
        d: (all_preds[all_true == d] == d).mean() for d in range(10)
    }
    cols = st.columns(10)
    for d, col in enumerate(cols):
        col.metric(str(d), f"{digit_accs[d]*100:.1f}%")

elif data_ok and not model_result:
    st.info("No model loaded. Use the sidebar to train one.")
