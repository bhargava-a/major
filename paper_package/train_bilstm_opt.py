"""
Dedicated trainer for the optimized BiLSTM + attention autoencoder.

Self-contained: contains the model, data loader, train loop, and plots.
Use this file to iterate on the architecture without touching the other 5
models.

Outputs (./results_bilstm_opt/)
  results.txt                      terminal log
  results.json                     per-epoch losses + run metadata
  plot_loss_curve.png              train/val curves (log scale)
  plot_recon_error_distribution.png  per-window test MSE histogram
  plot_error_timeline.png          per-window MSE vs window index
  plot_top_anomalies.png           highest-error test windows overlaid on
                                     their reconstruction

Usage
-----
  python train_bilstm_opt.py                     # defaults
  python train_bilstm_opt.py --epochs 80
  python train_bilstm_opt.py --d-model 96 --dropout 0.2
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cesnet_subset.npz")
    ap.add_argument("--out-dir", default="results_bilstm_opt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-epochs", type=int, default=8,
                    help="don't early-stop before this many epochs")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-lr-mult", type=float, default=1.5,
                    help="OneCycle peak = lr * this (was 3.0, too spiky)")
    ap.add_argument("--pct-start", type=float, default=0.2)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true",
                    help="disable AMP; recommended for smooth-descent runs")
    ap.add_argument("--no-sched", action="store_true")

    # model hyperparameters (tweak these to iterate on architecture)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead",   type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--ffn-mult", type=int, default=2)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# DUAL LOGGER
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8", buffering=1)
    def write(self, s):
        sys.__stdout__.write(s); sys.__stdout__.flush()
        self.f.write(s)
    def flush(self):
        sys.__stdout__.flush(); self.f.flush()
    def close(self):
        self.f.close()


def hr(c="-"):
    print(c * 78)


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

class BiLSTMAttnOptAE(nn.Module):
    """
    Optimized BiLSTM + attention autoencoder.

    Key design choices (tuned to beat plain transformer/tcn on this dataset):
      - Input projection to d_model + learnable positional embedding.
      - Single bidirectional LSTM layer; output dim = d_model.
      - Residual + LayerNorm around the BiLSTM (pre-norm style).
      - Multi-head self-attention block refines per-timestep encoding.
      - FFN block (GELU, 2x expansion) with residual + LN.
      - LSTM decoder at full per-timestep resolution (no pooling bottleneck).
      - Final residual skip from input to output so the model learns the
        delta from identity instead of recomputing the signal from scratch.
    """
    def __init__(self, n_features=18, d_model=64, nhead=4, window=10,
                 dropout=0.1, ffn_mult=2):
        super().__init__()
        self.in_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        self.bilstm = nn.LSTM(d_model, d_model // 2, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.ln3 = nn.LayerNorm(d_model)
        self.dec_lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.ln4 = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                 # [B, T, F]
        h0 = self.in_proj(x) + self.pos[:, :x.size(1)]    # [B, T, d]
        h, _ = self.bilstm(h0)
        h = self.ln1(h + h0)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.ln2(h + self.dropout(a))
        f = self.ffn(h)
        h = self.ln3(h + self.dropout(f))
        d, _ = self.dec_lstm(h)
        h = self.ln4(h + d)
        y = self.out(h)
        return y + x                                      # residual to input


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def build_per_entity_windows(data, entity_id, window):
    out = []
    for e in np.unique(entity_id):
        arr = data[entity_id == e]
        if len(arr) < window:
            continue
        for i in range(len(arr) - window + 1):
            out.append(arr[i:i + window])
    return np.stack(out, axis=0).astype(np.float32)


def load_splits(path, window):
    print(f"Loading {path} ...")
    z = np.load(path, allow_pickle=True)
    data, eid, split = z["data"], z["entity_id"], z["split"]
    feats = [str(x) for x in z["feature_names"]]
    print(f"  rows={len(data):,}  features={data.shape[1]}  "
          f"entities={np.unique(eid).size}")
    outs = {}
    for label, name in [(0, "train"), (1, "val"), (2, "test")]:
        m = split == label
        w = build_per_entity_windows(data[m], eid[m], window)
        outs[name] = torch.from_numpy(w)
        print(f"  {name:5s}: windows={len(w):,}  shape={w.shape}")
    return outs, feats


# ---------------------------------------------------------------------------
# TRAIN / EVAL
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        r = model(x)
        tot += F.mse_loss(r, x, reduction="sum").item()
        n += x.numel()
    return tot / max(1, n)


@torch.no_grad()
def per_window_errors(model, loader, device):
    model.eval()
    errs = []
    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        r = model(x)
        e = ((r - x) ** 2).mean(dim=(1, 2))
        errs.append(e.detach().cpu().numpy())
    return np.concatenate(errs) if errs else np.array([])


@torch.no_grad()
def measure_inference(model, loader, device, warm=2, iters=10):
    model.eval()
    x0 = next(iter(loader))[0].to(device)
    for _ in range(warm): _ = model(x0)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): _ = model(x0)
    if device.type == "cuda": torch.cuda.synchronize()
    total = time.perf_counter() - t0
    per_batch_ms = total / iters * 1000
    return per_batch_ms, per_batch_ms / x0.size(0)


def train(model, loaders, args, device):
    train_loader, val_loader, test_loader = loaders
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    crit = nn.MSELoss()

    sched = None
    if not args.no_sched:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr * args.max_lr_mult,
            steps_per_epoch=len(train_loader), epochs=args.epochs,
            pct_start=args.pct_start, anneal_strategy="cos",
            div_factor=10.0, final_div_factor=1e3,
        )

    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP: {use_amp}   scheduler: {'OneCycleLR' if sched else 'off'}")

    best_val = float("inf"); best_state = None
    patience_left = args.patience
    th, vh = [], []

    t_start = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tot, cnt = 0.0, 0
        for (x,) in train_loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                r = model(x)
                loss = crit(r, x)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if sched is not None: sched.step()
            tot += loss.item() * x.size(0); cnt += x.size(0)
        tr = tot / cnt
        va = evaluate(model, val_loader, device)
        th.append(tr); vh.append(va)
        dt = time.perf_counter() - t0

        tag = ""
        if va < best_val - 1e-9:
            best_val = va
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_left = args.patience
            tag = "  <- best"
        else:
            if ep >= args.min_epochs:
                patience_left -= 1
            tag = f"  (patience {patience_left}/{args.patience})"
        print(f"epoch {ep:3d}/{args.epochs}  "
              f"train={tr:.6f}  val={va:.6f}  time={dt:.1f}s{tag}")
        if patience_left <= 0 and ep >= args.min_epochs:
            print(f"Early stop at epoch {ep}")
            break

    total_time = time.perf_counter() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = evaluate(model, test_loader, device)
    ib, isa = measure_inference(model, test_loader, device)
    errs = per_window_errors(model, test_loader, device)

    return {
        "epochs_ran": len(th),
        "train_time_sec": total_time,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "infer_ms_per_batch": ib,
        "infer_us_per_sample": isa * 1000,
        "train_history": th,
        "val_history": vh,
        "test_errors": errs,
    }


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def plot_loss_curve(th, vh, out_dir, best_val):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(th) + 1), th, label="train", lw=2, color="#4C72B0")
    ax.plot(range(1, len(vh) + 1), vh, label="val",   lw=2, color="#C44E52", ls="--")
    ax.set_yscale("log"); ax.grid(alpha=0.3)
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (log)")
    ax.set_title(f"bilstm_attn_opt  (best val={best_val:.6f})")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "plot_loss_curve.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"saved {p}")


def plot_error_distribution(errs, out_dir):
    e = np.log10(np.clip(errs, 1e-12, None))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(e, bins=80, color="#8172B2", alpha=0.85, density=True)
    p95 = np.percentile(e, 95); p99 = np.percentile(e, 99)
    ax.axvline(p95, color="orange", ls="--", label=f"p95 = 10^{p95:.2f}")
    ax.axvline(p99, color="red",    ls="--", label=f"p99 = 10^{p99:.2f}")
    ax.set_xlabel("log10(per-window MSE)")
    ax.set_ylabel("density")
    ax.set_title("Test-set reconstruction error distribution")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "plot_recon_error_distribution.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"saved {p}")


def plot_error_timeline(errs, out_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(errs, lw=0.5, color="#4C72B0")
    thr95 = np.percentile(errs, 95); thr99 = np.percentile(errs, 99)
    mask95 = errs > thr95; mask99 = errs > thr99
    ax.scatter(np.where(mask95)[0], errs[mask95], s=6, color="orange",
               label=f">p95 ({mask95.sum()})")
    ax.scatter(np.where(mask99)[0], errs[mask99], s=10, color="red",
               label=f">p99 ({mask99.sum()})")
    ax.set_yscale("log")
    ax.set_xlabel("test window index")
    ax.set_ylabel("per-window MSE (log)")
    ax.set_title("Reconstruction error over test set")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "plot_error_timeline.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"saved {p}")


def plot_top_anomalies(model, test_loader, device, feats, out_dir, k=4):
    """For the k highest-error test windows, overlay input vs reconstruction."""
    model.eval()
    xs, rs, errs = [], [], []
    with torch.no_grad():
        for (x,) in test_loader:
            x = x.to(device)
            r = model(x)
            e = ((r - x) ** 2).mean(dim=(1, 2))
            xs.append(x.cpu().numpy()); rs.append(r.cpu().numpy())
            errs.append(e.cpu().numpy())
    x = np.concatenate(xs); r = np.concatenate(rs); e = np.concatenate(errs)
    top = np.argsort(-e)[:k]

    fig, axes = plt.subplots(k, 1, figsize=(10, 2.5 * k), sharex=True)
    if k == 1: axes = [axes]
    show = ["n_flows", "n_packets", "n_bytes"]
    show_idx = [feats.index(s) for s in show if s in feats]
    for ax, idx in zip(axes, top):
        for si, sname in zip(show_idx, show):
            ax.plot(x[idx, :, si], lw=1.5, label=f"{sname} (in)")
            ax.plot(r[idx, :, si], lw=1.5, ls="--", label=f"{sname} (rec)")
        ax.set_title(f"window {idx}  MSE={e[idx]:.5f}", fontsize=9)
        ax.grid(alpha=0.3)
        if idx == top[0]: ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("t in window")
    fig.suptitle("Top-k anomalous windows: input vs reconstruction", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(out_dir, "plot_top_anomalies.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"saved {p}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.out_dir, "results.txt"))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hr("=")
    print("bilstm_attn_opt - dedicated training script")
    hr("=")
    print(f"device    : {device}  "
          f"({torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'})")
    print(f"epochs    : {args.epochs}")
    print(f"patience  : {args.patience}")
    print(f"batch size: {args.batch_size}")
    print(f"lr        : {args.lr}  (OneCycle peak = {args.lr*args.max_lr_mult:.2e})")
    print(f"pct_start : {args.pct_start}")
    print(f"min_epochs: {args.min_epochs}")
    print(f"window    : {args.window}")
    print(f"d_model   : {args.d_model}")
    print(f"nhead     : {args.nhead}")
    print(f"dropout   : {args.dropout}")
    print(f"ffn_mult  : {args.ffn_mult}")
    print(f"seed      : {args.seed}")

    splits, feats = load_splits(args.data, args.window)
    n_features = splits["train"].shape[2]

    pin = device.type == "cuda"
    train_loader = DataLoader(TensorDataset(splits["train"]),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(TensorDataset(splits["val"]),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(TensorDataset(splits["test"]),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    model = BiLSTMAttnOptAE(
        n_features=n_features,
        d_model=args.d_model,
        nhead=args.nhead,
        window=args.window,
        dropout=args.dropout,
        ffn_mult=args.ffn_mult,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hr("=")
    print(f"MODEL params: {n_params:,}")
    hr("=")

    res = train(model, (train_loader, val_loader, test_loader), args, device)

    hr("=")
    print("RESULTS")
    hr("=")
    print(f"params              : {n_params:,}")
    print(f"epochs ran          : {res['epochs_ran']}")
    print(f"total train time    : {res['train_time_sec']:.1f} s")
    print(f"best val MSE        : {res['best_val_loss']:.6f}")
    print(f"test MSE            : {res['test_loss']:.6f}")
    print(f"inference ms/batch  : {res['infer_ms_per_batch']:.2f}")
    print(f"inference us/sample : {res['infer_us_per_sample']:.2f}")

    # save JSON (keep it small - do not dump per-window errors)
    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "params": n_params,
            "epochs_ran": res["epochs_ran"],
            "train_time_sec": res["train_time_sec"],
            "best_val_loss": res["best_val_loss"],
            "test_loss": res["test_loss"],
            "infer_ms_per_batch": res["infer_ms_per_batch"],
            "infer_us_per_sample": res["infer_us_per_sample"],
            "train_history": res["train_history"],
            "val_history":   res["val_history"],
            "test_errors_n": len(res["test_errors"]),
        }, f, indent=2)
    print(f"saved {json_path}")

    hr("=")
    print("PLOTS")
    hr("=")
    plot_loss_curve(res["train_history"], res["val_history"],
                    args.out_dir, res["best_val_loss"])
    plot_error_distribution(res["test_errors"], args.out_dir)
    plot_error_timeline(res["test_errors"], args.out_dir)
    plot_top_anomalies(model, test_loader, device, feats, args.out_dir, k=4)

    print(f"\nAll outputs in: {args.out_dir}/")


if __name__ == "__main__":
    main()
