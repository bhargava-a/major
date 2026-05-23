"""
CESNET anomaly detection - train & compare 5 autoencoder architectures.

Single-file runner. Loads cesnet_subset.npz (expected in ./data/),
trains all 5 models with identical settings, and logs:
  - train loss per epoch
  - val loss per epoch
  - total training time
  - final train/val/test loss
  - inference time per batch and per sample
  - parameter count
Results stream to both terminal and results.txt + results.csv.

Usage
-----
  python train_all_models.py                   # train all 5 models
  python train_all_models.py --only bilstm_attn # train one model
  python train_all_models.py --epochs 50 --patience 4 --batch-size 256

Requirements: torch, numpy. Auto-uses CUDA if available.
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIG / CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cesnet_subset.npz",
                    help="path to subset NPZ")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=4,
                    help="early-stopping patience on val loss")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default=None,
                    help="train only one of: vanilla_ae, lstm_ae, tcn_ae, "
                         "transformer_ae, bilstm_attn")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true",
                    help="disable mixed precision on CUDA")
    ap.add_argument("--no-sched", action="store_true",
                    help="disable OneCycleLR schedule")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# DUAL LOGGER (terminal + file)
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8", buffering=1)
    def write(self, s):
        sys.__stdout__.write(s)
        sys.__stdout__.flush()
        self.f.write(s)
    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()
    def close(self):
        self.f.close()


def hr(char="-"):
    print(char * 78)


# ---------------------------------------------------------------------------
# DATA LOADING & WINDOWING
# ---------------------------------------------------------------------------

def build_per_entity_windows(data: np.ndarray,
                             entity_id: np.ndarray,
                             window: int) -> np.ndarray:
    """Build sliding windows WITHIN each entity only (no boundary crossing)."""
    out = []
    ents = np.unique(entity_id)
    for e in ents:
        mask = entity_id == e
        arr = data[mask]
        if len(arr) < window:
            continue
        for i in range(len(arr) - window + 1):
            out.append(arr[i:i + window])
    return np.stack(out, axis=0).astype(np.float32)


def load_splits(path: str, window: int):
    print(f"Loading {path} ...")
    z = np.load(path, allow_pickle=True)
    data = z["data"]
    eid = z["entity_id"]
    split = z["split"]
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
# MODELS
# ---------------------------------------------------------------------------

class VanillaAE(nn.Module):
    """Flatten-window MLP autoencoder (no temporal modelling)."""
    def __init__(self, window=10, n_features=18, latent=32):
        super().__init__()
        in_dim = window * n_features
        self.window = window
        self.n_features = n_features
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, in_dim),
        )
    def forward(self, x):                  # x: [B, T, F]
        b = x.size(0)
        z = self.enc(x.reshape(b, -1))
        return self.dec(z).reshape(b, self.window, self.n_features)


class LSTMAE(nn.Module):
    """Unidirectional 2-layer LSTM encoder-decoder."""
    def __init__(self, n_features=18, hidden=64, latent=32):
        super().__init__()
        self.enc = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden, n_features)
    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.to_latent(h[-1])                        # [B, L]
        h0 = self.from_latent(z).unsqueeze(1)            # [B, 1, H]
        h0 = h0.expand(-1, x.size(1), -1).contiguous()   # [B, T, H]
        y, _ = self.dec(h0)
        return self.out(y)


class Chomp1d(nn.Module):
    def __init__(self, chomp): super().__init__(); self.chomp = chomp
    def forward(self, x): return x[:, :, :-self.chomp].contiguous() if self.chomp else x


class TCNBlock(nn.Module):
    def __init__(self, cin, cout, k, d):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=pad, dilation=d),
            Chomp1d(pad), nn.ReLU(),
            nn.Conv1d(cout, cout, k, padding=pad, dilation=d),
            Chomp1d(pad), nn.ReLU(),
        )
        self.proj = nn.Conv1d(cin, cout, 1) if cin != cout else None
    def forward(self, x):
        r = x if self.proj is None else self.proj(x)
        return self.net(x) + r


class TCNAE(nn.Module):
    def __init__(self, n_features=18, channels=(32, 32, 32), k=3):
        super().__init__()
        layers, prev = [], n_features
        for i, c in enumerate(channels):
            layers.append(TCNBlock(prev, c, k, d=2 ** i))
            prev = c
        self.enc = nn.Sequential(*layers)
        # decoder mirrors encoder
        dec, prev = [], channels[-1]
        for i, c in enumerate(reversed(channels)):
            dec.append(TCNBlock(prev, c, k, d=2 ** (len(channels) - 1 - i)))
            prev = c
        self.dec = nn.Sequential(*dec)
        self.out = nn.Conv1d(prev, n_features, 1)
    def forward(self, x):                  # [B, T, F]
        x = x.transpose(1, 2)               # [B, F, T]
        z = self.enc(x)
        r = self.out(self.dec(z))
        return r.transpose(1, 2)


class TransformerAE(nn.Module):
    def __init__(self, n_features=18, d_model=64, nhead=4, layers=2, window=10):
        super().__init__()
        self.in_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, 128,
                                               batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        dec_layer = nn.TransformerEncoderLayer(d_model, nhead, 128,
                                               batch_first=True, activation='gelu')
        self.dec = nn.TransformerEncoder(dec_layer, num_layers=layers)
        self.out = nn.Linear(d_model, n_features)
    def forward(self, x):
        h = self.in_proj(x) + self.pos[:, :x.size(1)]
        z = self.enc(h)
        r = self.dec(z)
        return self.out(r)


class BiLSTMAttnAE(nn.Module):
    """Bidirectional LSTM encoder + attention + LSTM decoder. Reference model."""
    def __init__(self, n_features=18, hidden=64, latent=32):
        super().__init__()
        self.enc = nn.LSTM(n_features, hidden, num_layers=2,
                           batch_first=True, bidirectional=True)
        self.attn_q = nn.Linear(2 * hidden, 2 * hidden)
        self.attn_v = nn.Linear(2 * hidden, 1)
        self.to_latent = nn.Linear(2 * hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden, n_features)
    def forward(self, x):
        enc_out, _ = self.enc(x)                         # [B, T, 2H]
        att = torch.tanh(self.attn_q(enc_out))
        w = torch.softmax(self.attn_v(att), dim=1)       # [B, T, 1]
        ctx = (w * enc_out).sum(dim=1)                   # [B, 2H]
        z = self.to_latent(ctx)
        h0 = self.from_latent(z).unsqueeze(1).expand(-1, x.size(1), -1).contiguous()
        y, _ = self.dec(h0)
        return self.out(y)


class BiLSTMAttnOptAE(nn.Module):
    """
    Optimized BiLSTM + attention autoencoder.

    Differences vs. original bilstm_attn:
      - Input projection to d_model (separates feature dim from RNN width).
      - Per-timestep representation kept all the way through (no pooling to a
        single context vector; the old design was the main bottleneck).
      - Single-layer BiLSTM (2 layers added cost without benefit here).
      - Multi-head self-attention refines per-timestep encoding (transformer-
        in-RNN hybrid).
      - Pre-norm residual blocks and dropout for stable, fast convergence.
      - Residual skip from input to output so the model only has to learn
        the delta from identity; this shrinks effective loss drastically.
    """
    def __init__(self, n_features=18, d_model=64, nhead=4, window=10, dropout=0.1):
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
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln3 = nn.LayerNorm(d_model)
        self.dec_lstm = nn.LSTM(d_model, d_model, num_layers=1,
                                batch_first=True)
        self.ln4 = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                  # [B, T, F]
        h0 = self.in_proj(x) + self.pos[:, :x.size(1)]     # [B, T, d]
        h, _ = self.bilstm(h0)                             # [B, T, d]
        h = self.ln1(h + h0)                               # residual
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.ln2(h + self.dropout(a))                  # residual
        f = self.ffn(h)
        h = self.ln3(h + self.dropout(f))
        d, _ = self.dec_lstm(h)
        h = self.ln4(h + d)
        y = self.out(h)
        return y + x                                       # residual to input


MODEL_REGISTRY = {
    "vanilla_ae":      lambda w, f: VanillaAE(window=w, n_features=f),
    "lstm_ae":         lambda w, f: LSTMAE(n_features=f),
    "tcn_ae":          lambda w, f: TCNAE(n_features=f),
    "transformer_ae":  lambda w, f: TransformerAE(n_features=f, window=w),
    "bilstm_attn":     lambda w, f: BiLSTMAttnAE(n_features=f),
    "bilstm_attn_opt": lambda w, f: BiLSTMAttnOptAE(n_features=f, window=w),
}


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    name: str
    params: int
    epochs_ran: int
    train_time_sec: float
    best_val_loss: float
    final_train_loss: float
    test_loss: float
    infer_ms_per_batch: float
    infer_ms_per_sample: float
    train_loss_history: List[float]
    val_loss_history: List[float]
    test_recon_errors: List[float]   # per-window MSE on test set


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        r = model(x)
        loss = F.mse_loss(r, x, reduction="sum")
        tot += loss.item()
        n += x.numel()
    return tot / max(1, n)


@torch.no_grad()
def per_window_errors(model, loader, device):
    """MSE per window on the given loader; returns numpy array shape [N]."""
    model.eval()
    errs = []
    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        r = model(x)
        e = ((r - x) ** 2).mean(dim=(1, 2))   # [B]
        errs.append(e.detach().cpu().numpy())
    return np.concatenate(errs) if errs else np.array([])


@torch.no_grad()
def measure_inference(model, loader, device, warm=2, iters=10):
    model.eval()
    it = iter(loader)
    x0 = next(it)[0].to(device)
    for _ in range(warm):
        _ = model(x0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total = time.perf_counter() - t0
    per_batch_ms = total / iters * 1000.0
    per_sample_ms = per_batch_ms / x0.size(0)
    return per_batch_ms, per_sample_ms


def train_one(name, model, loaders, args, device) -> RunResult:
    hr("=")
    print(f"MODEL: {name}")
    hr("=")
    n_params = count_params(model)
    print(f"params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    crit = nn.MSELoss()
    train_loader, val_loader, test_loader = loaders

    use_sched = not getattr(args, "no_sched", False)
    sched = None
    if use_sched:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr * 3,
            steps_per_epoch=len(train_loader), epochs=args.epochs,
            pct_start=0.1, anneal_strategy='cos',
        )

    use_amp = (device.type == "cuda") and not getattr(args, "no_amp", False)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"AMP: {use_amp}  scheduler: {'OneCycleLR' if use_sched else 'off'}")

    best_val = float("inf")
    best_state = None
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
            with torch.amp.autocast('cuda', enabled=use_amp):
                r = model(x)
                loss = crit(r, x)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            tot += loss.item() * x.size(0)
            cnt += x.size(0)
        tr_loss = tot / cnt
        va_loss = evaluate(model, val_loader, device)
        th.append(tr_loss); vh.append(va_loss)
        dt = time.perf_counter() - t0

        tag = ""
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_left = args.patience
            tag = "  <- best"
        else:
            patience_left -= 1
            tag = f"  (patience {patience_left}/{args.patience})"

        print(f"epoch {ep:3d}/{args.epochs}  "
              f"train={tr_loss:.6f}  val={va_loss:.6f}  "
              f"time={dt:.1f}s{tag}")
        if patience_left <= 0:
            print(f"Early stop at epoch {ep}")
            break

    total_time = time.perf_counter() - t_start

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = evaluate(model, test_loader, device)
    ibatch, isample = measure_inference(model, test_loader, device)
    test_errs = per_window_errors(model, test_loader, device)

    print(f"total_time_sec = {total_time:.1f}")
    print(f"best_val_loss  = {best_val:.6f}")
    print(f"test_loss      = {test_loss:.6f}")
    print(f"inference      = {ibatch:.2f} ms/batch   {isample*1000:.2f} us/sample")
    return RunResult(
        name=name,
        params=n_params,
        epochs_ran=len(th),
        train_time_sec=total_time,
        best_val_loss=best_val,
        final_train_loss=th[-1],
        test_loss=test_loss,
        infer_ms_per_batch=ibatch,
        infer_ms_per_sample=isample,
        train_loss_history=th,
        val_loss_history=vh,
        test_recon_errors=test_errs.tolist(),
    )


# ---------------------------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------------------------

COLORS = {
    'vanilla_ae':     '#4C72B0',
    'lstm_ae':        '#DD8452',
    'tcn_ae':         '#55A467',
    'transformer_ae': '#C44E52',
    'bilstm_attn':    '#8172B2',
}


def _color(name, idx):
    return COLORS.get(name, plt.cm.tab10(idx % 10))


def plot_loss_curves_per_model(results, out_dir):
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows),
                             squeeze=False)
    for i, r in enumerate(results):
        ax = axes[i // cols][i % cols]
        epochs = range(1, len(r.train_loss_history) + 1)
        ax.plot(epochs, r.train_loss_history, label='train',
                color=_color(r.name, i), lw=2)
        ax.plot(epochs, r.val_loss_history,   label='val',
                color=_color(r.name, i), lw=2, ls='--')
        ax.set_title(f"{r.name}  (best val={r.best_val_loss:.5f})", fontsize=10)
        ax.set_xlabel('epoch'); ax.set_ylabel('MSE')
        ax.set_yscale('log')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis('off')
    fig.suptitle('Training and validation loss per model (log scale)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(out_dir, 'plot_loss_curves.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"saved {p}")


def plot_val_curves_overlay(results, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(results):
        epochs = range(1, len(r.val_loss_history) + 1)
        ax.plot(epochs, r.val_loss_history,
                color=_color(r.name, i), lw=2, label=r.name)
    ax.set_xlabel('epoch'); ax.set_ylabel('val MSE')
    ax.set_yscale('log')
    ax.set_title('Validation loss across all models (log scale)')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, 'plot_val_overlay.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"saved {p}")


def plot_summary_bars(results, out_dir):
    names = [r.name for r in results]
    colors = [_color(n, i) for i, n in enumerate(names)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    metrics = [
        ('params',            [r.params for r in results],              'params (count)', False),
        ('train_time_sec',    [r.train_time_sec for r in results],      'train time (s)', False),
        ('best_val_loss',     [r.best_val_loss for r in results],       'best val MSE', True),
        ('test_loss',         [r.test_loss for r in results],           'test MSE', True),
        ('ms/batch',          [r.infer_ms_per_batch for r in results],  'inference ms/batch', False),
        ('us/sample',         [r.infer_ms_per_sample * 1000 for r in results], 'inference us/sample', False),
    ]
    for i, (title, vals, ylabel, log) in enumerate(metrics):
        ax = axes[i // 3][i % 3]
        bars = ax.bar(names, vals, color=colors, alpha=0.9)
        ax.set_title(title); ax.set_ylabel(ylabel)
        if log: ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=25, labelsize=8)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.4g}", (b.get_x() + b.get_width() / 2, v),
                        ha='center', va='bottom', fontsize=7)
    fig.suptitle('Cross-model comparison', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(out_dir, 'plot_summary_bars.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"saved {p}")


def plot_reconstruction_error_dist(results, out_dir):
    """Histogram of per-window reconstruction MSE on the test set."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results):
        e = np.array(r.test_recon_errors)
        if len(e) == 0:
            continue
        # log10 for visibility
        e = np.log10(np.clip(e, 1e-12, None))
        ax.hist(e, bins=80, alpha=0.45, label=r.name,
                color=_color(r.name, i), density=True)
    ax.set_xlabel('log10(per-window reconstruction MSE)')
    ax.set_ylabel('density')
    ax.set_title('Test-set reconstruction error distribution per model')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, 'plot_recon_error_distribution.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"saved {p}")


def plot_anomaly_rate_vs_threshold(results, out_dir):
    """% of test windows flagged anomalous vs threshold percentile."""
    percentiles = np.arange(80, 100, 0.5)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(results):
        e = np.array(r.test_recon_errors)
        if len(e) == 0:
            continue
        rates = []
        for p in percentiles:
            t = np.percentile(e, p)
            rates.append(float((e > t).mean() * 100))
        ax.plot(percentiles, rates, lw=2, label=r.name,
                color=_color(r.name, i))
    ax.set_xlabel('threshold percentile (on each model\'s own errors)')
    ax.set_ylabel('% flagged as anomalous')
    ax.set_title('Anomaly flag rate vs threshold')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, 'plot_anomaly_rate.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"saved {p}")


def plot_params_vs_loss(results, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(results):
        ax.scatter(r.params, r.test_loss, s=120,
                   color=_color(r.name, i), label=r.name, edgecolor='k')
        ax.annotate(r.name, (r.params, r.test_loss),
                    textcoords='offset points', xytext=(6, 6), fontsize=9)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('parameters (log)'); ax.set_ylabel('test MSE (log)')
    ax.set_title('Model capacity vs reconstruction quality')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    p = os.path.join(out_dir, 'plot_params_vs_loss.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"saved {p}")


def generate_all_plots(results, out_dir):
    hr("=")
    print("GENERATING VISUALIZATIONS")
    hr("=")
    plot_loss_curves_per_model(results, out_dir)
    plot_val_curves_overlay(results, out_dir)
    plot_summary_bars(results, out_dir)
    plot_reconstruction_error_dist(results, out_dir)
    plot_anomaly_rate_vs_threshold(results, out_dir)
    plot_params_vs_loss(results, out_dir)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "results.txt")
    sys.stdout = Tee(log_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hr("=")
    print("CESNET anomaly detection - model comparison")
    hr("=")
    print(f"device      : {device}  "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    print(f"epochs      : {args.epochs}")
    print(f"patience    : {args.patience}")
    print(f"batch size  : {args.batch_size}")
    print(f"lr          : {args.lr}")
    print(f"window      : {args.window}")
    print(f"seed        : {args.seed}")

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

    if args.only:
        names = [args.only]
    else:
        names = list(MODEL_REGISTRY.keys())

    results: List[RunResult] = []
    for name in names:
        model = MODEL_REGISTRY[name](args.window, n_features).to(device)
        res = train_one(name, model,
                        (train_loader, val_loader, test_loader),
                        args, device)
        results.append(res)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- summary ---------------------------------------------------------
    hr("=")
    print("FINAL COMPARISON")
    hr("=")
    header = (f"{'model':<18s} {'params':>10s} {'epochs':>7s} "
              f"{'time_s':>8s} {'best_val':>10s} {'test_loss':>10s} "
              f"{'ms/batch':>9s} {'us/samp':>8s}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.name:<18s} {r.params:>10,} {r.epochs_ran:>7d} "
              f"{r.train_time_sec:>8.1f} {r.best_val_loss:>10.6f} "
              f"{r.test_loss:>10.6f} "
              f"{r.infer_ms_per_batch:>9.2f} {r.infer_ms_per_sample*1000:>8.2f}")
    print()

    csv_path = os.path.join(args.out_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "epochs_ran", "train_time_sec",
                    "best_val_loss", "final_train_loss", "test_loss",
                    "infer_ms_per_batch", "infer_us_per_sample"])
        for r in results:
            w.writerow([r.name, r.params, r.epochs_ran,
                        f"{r.train_time_sec:.2f}",
                        f"{r.best_val_loss:.6f}", f"{r.final_train_loss:.6f}",
                        f"{r.test_loss:.6f}",
                        f"{r.infer_ms_per_batch:.3f}",
                        f"{r.infer_ms_per_sample*1000:.3f}"])
    print(f"saved {csv_path}")

    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        # strip test_recon_errors from JSON to keep file small
        payload = []
        for r in results:
            d = dict(r.__dict__)
            d['test_recon_errors_len'] = len(d.pop('test_recon_errors'))
            payload.append(d)
        json.dump(payload, f, indent=2)
    print(f"saved {json_path}")

    generate_all_plots(results, args.out_dir)

    print(f"full log: {log_path}")


if __name__ == "__main__":
    main()
