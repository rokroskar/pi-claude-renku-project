import marimo

__generated_with = "0.13.0"
app = marimo.App(width="wide", title="MNIST Digit Classifier")


@app.cell
def _():
    import os
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from model import CNN, load_mnist
    return CNN, DataLoader, Path, TensorDataset, load_mnist, mo, np, os, plt, subprocess, sys, torch


@app.cell
def _(Path, os, torch):
    DATA_DIR = Path(os.environ.get("MNIST_DATA_DIR", "/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130"))
    MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/home/renku/work/model-artifacts/mnist-cnn-claude"))
    MODEL_PATH = MODEL_DIR / "mnist_cnn.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FALLBACK_PATHS = [
        Path("/home/renku/work/model-artifacts/mnist-models/mnist_cnn.pt"),
        Path("/home/renku/work/pretrained-model-artifacts/mnist_cnn.pt"),
        Path("/home/renku/work/pretrained-model-artifacts/mnist-models/mnist_cnn.pt"),
    ]
    return DATA_DIR, DEVICE, FALLBACK_PATHS, MODEL_DIR, MODEL_PATH


@app.cell
def _(CNN, DEVICE, FALLBACK_PATHS, MODEL_PATH, torch):
    def _find():
        for p in [MODEL_PATH] + FALLBACK_PATHS:
            if p.exists():
                ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
                m = CNN().to(DEVICE)
                m.load_state_dict(ckpt["model_state"])
                m.eval()
                return m, ckpt.get("accuracy", "?"), ckpt.get("epoch", "?"), str(p)
        return None

    model_result = _find()
    return (model_result,)


@app.cell
def _(DATA_DIR, load_mnist, np):
    _, _, _x, _y = load_mnist(DATA_DIR)
    x_test = np.array(_x, dtype=np.float32)
    if x_test.max() > 1.5:
        x_test = x_test / 255.0
    y_test = np.array(_y, dtype=np.int64)
    return x_test, y_test


@app.cell
def _(mo):
    mo.md("# MNIST Digit Classifier")


@app.cell
def _(mo, model_result):
    if model_result:
        _, acc, epoch, model_path = model_result
        mo.callout(mo.md(f"**Model loaded** — accuracy **{acc:.4f}**, epoch {epoch}  \n`{model_path}`"), kind="success")
    else:
        mo.callout(mo.md("No trained model found."), kind="warn")


@app.cell
def _(mo):
    n_cols = mo.ui.slider(4, 10, value=6, label="Columns")
    n_rows = mo.ui.slider(2, 6, value=3, label="Rows")
    shuffle_btn = mo.ui.run_button(label="🔀 Shuffle")
    return n_cols, n_rows, shuffle_btn


@app.cell
def _(mo, n_cols, n_rows, shuffle_btn):
    mo.hstack([n_cols, n_rows, shuffle_btn], gap=2)


@app.cell
def _(DEVICE, model_result, mo, n_cols, n_rows, np, plt, shuffle_btn, torch, x_test, y_test):
    mo.stop(model_result is None)
    _shuffle = shuffle_btn.value  # reactive on each click

    model, _acc, _epoch, _path = model_result
    _n = n_cols.value * n_rows.value
    _idxs = np.random.choice(len(x_test), _n, replace=False)
    _imgs = x_test[_idxs]
    _labels = y_test[_idxs]

    _t = torch.tensor(_imgs).unsqueeze(1) if _imgs.ndim == 3 else torch.tensor(_imgs)
    with torch.no_grad():
        _logits = model(_t.to(DEVICE))
        _probs = torch.softmax(_logits, dim=1).cpu().numpy()
        _preds = _logits.argmax(1).cpu().numpy()

    _fig, _axes = plt.subplots(n_rows.value, n_cols.value, figsize=(n_cols.value * 1.6, n_rows.value * 2.0))
    for _ax, _img, _lbl, _pred, _prob in zip(np.array(_axes).flat, _imgs, _labels, _preds, _probs):
        _ax.imshow(_img.squeeze(), cmap="gray")
        _color = "green" if _pred == _lbl else "red"
        _ax.set_title(f"pred={_pred}\ntrue={_lbl}\n{_prob[_pred]*100:.1f}%", fontsize=7, color=_color)
        _ax.axis("off")
    plt.tight_layout()
    _fig


@app.cell
def _(mo):
    mo.md("## Per-digit accuracy on full test set")


@app.cell
def _(DEVICE, DataLoader, TensorDataset, model_result, mo, np, torch, x_test, y_test):
    mo.stop(model_result is None)
    model_digit, _, _, _ = model_result

    _ds = TensorDataset(
        torch.tensor(x_test).unsqueeze(1) if x_test.ndim == 3 else torch.tensor(x_test),
        torch.tensor(y_test),
    )
    _loader = DataLoader(_ds, batch_size=512, num_workers=0)
    _preds_all, _true_all = [], []
    model_digit.eval()
    with torch.no_grad():
        for _xb, _yb in _loader:
            _preds_all.append(model_digit(_xb.to(DEVICE)).argmax(1).cpu().numpy())
            _true_all.append(_yb.numpy())
    _preds_all = np.concatenate(_preds_all)
    _true_all = np.concatenate(_true_all)
    _accs = {d: (_preds_all[_true_all == d] == d).mean() for d in range(10)}

    mo.hstack(
        [mo.stat(value=f"{_accs[d]*100:.1f}%", label=str(d)) for d in range(10)],
        gap=1,
    )


@app.cell
def _(mo):
    mo.md("## Re-train model")


@app.cell
def _(mo):
    target_acc = mo.ui.slider(0.95, 0.999, value=0.99, step=0.001, label="Target accuracy", show_value=True)
    max_epochs = mo.ui.number(5, 100, value=30, label="Max epochs")
    train_btn = mo.ui.run_button(label="▶ Start training")
    return max_epochs, target_acc, train_btn


@app.cell
def _(mo, target_acc, max_epochs, train_btn):
    mo.hstack([target_acc, max_epochs, train_btn], gap=2)


@app.cell
def _(DATA_DIR, MODEL_DIR, max_epochs, mo, os, subprocess, sys, target_acc, train_btn):
    mo.stop(not train_btn.value)
    _env = os.environ.copy()
    _env["TARGET_ACCURACY"] = str(target_acc.value)
    _env["MAX_EPOCHS"] = str(int(max_epochs.value))
    _env["MNIST_DATA_DIR"] = str(DATA_DIR)
    _env["MODEL_DIR"] = str(MODEL_DIR)
    _repo = Path("/home/renku/work/pi-claude-renku-project")
    _train = _repo / "train.py" if _repo.exists() else Path(__file__).parent / "train.py"
    with mo.status.spinner("Training…"):
        _proc = subprocess.run(
            [sys.executable, str(_train)],
            env=_env, capture_output=True, text=True,
        )
    if _proc.returncode == 0:
        mo.callout(mo.md("**Training complete!** Reload the page to use the new model."), kind="success")
    else:
        mo.callout(mo.md(f"**Training failed.**\n```\n{_proc.stdout[-2000:]}\n{_proc.stderr[-500:]}\n```"), kind="danger")


if __name__ == "__main__":
    app.run()
