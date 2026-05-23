"""
MASTER TRAINING & COMPARISON SCRIPT
====================================
Trains all 5 models sequentially with 80/20 train/val split.
Runs Isolation Forest + One-Class SVM baselines.
Prints two publication-ready comparison tables covering:
  - Architecture properties & computational complexity
  - Detection performance & score distribution
  - Convergence behaviour (train loss, val loss, gap)

Usage: python master_train_compare.py
"""

import os, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed — saves directly to PNG
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
NPZ_PATH   = r"c:\Users\bharg\OneDrive\Documents\GPU_package\data\cesnet_10min.npz"
OUT_DIR    = r"c:\Users\bharg\OneDrive\Documents\GPU_package\results"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBSAMPLE  = 3_000_000     # rows from 18M
WINDOW     = 10
EPOCHS     = 20
LR         = 1e-3
VAL_SPLIT  = 0.20
SEED       = 42
THRESHOLD_PERCENTILE = 95

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── GPU optimisations (Ampere / RTX 30xx) ─────────────────────────────────────
torch.backends.cudnn.benchmark        = True   # auto-find fastest conv algo
torch.backends.cuda.matmul.allow_tf32 = True   # TF32 matmul on Ampere (~30% faster)
torch.backends.cudnn.allow_tf32       = True   # TF32 for cuDNN ops
# ──────────────────────────────────────────────────────────────────────────────

print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"TF32   : enabled  |  cudnn.benchmark: enabled\n")

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
raw_full = np.load(NPZ_PATH)['data']
idx = np.random.default_rng(SEED).choice(len(raw_full), SUBSAMPLE, replace=False)
idx.sort()                              # preserve temporal order
raw = raw_full[idx].astype(np.float32)
print(f"Subsampled: {len(raw):,} rows from {len(raw_full):,}\n")

class FlatDataset(Dataset):
    """For Vanilla AE — no windowing."""
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self):  return len(self.data)
    def __getitem__(self, i): return self.data[i]

class WindowDataset(Dataset):
    """On-the-fly sliding window — avoids storing 22M windows in RAM."""
    def __init__(self, data, window):
        self.data   = torch.tensor(data, dtype=torch.float32)
        self.window = window
    def __len__(self):
        return len(self.data) - self.window + 1
    def __getitem__(self, i):
        return self.data[i:i + self.window]

def make_loaders(dataset, batch_train, batch_eval=2048):
    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_train,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_eval,
                              shuffle=False, num_workers=0, pin_memory=True)
    full_loader  = DataLoader(dataset,  batch_size=batch_eval,
                              shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, full_loader, n_train, n_val

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

EARLY_STOP_PATIENCE = 5      # stop if val_loss doesn't improve for 5 epochs
EARLY_STOP_MIN_DELTA = 1e-7  # minimum improvement to count as progress

def train_model(model, train_loader, val_loader, n_train, n_val, label):
    # torch.compile — fuses ops, removes Python overhead (~10-20% speedup)
    if DEVICE.type == "cuda":
        try:
            model = torch.compile(model)
        except Exception:
            pass   # compile not available on this platform, skip silently

    optimizer    = torch.optim.Adam(model.parameters(), lr=LR)
    criterion    = nn.MSELoss()
    scaler       = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    # OneCycleLR — ramps LR up then down, converges in ~half the epochs
    scheduler    = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer, max_lr=LR * 10,
                        steps_per_epoch=len(train_loader),
                        epochs=EPOCHS, pct_start=0.3)
    history      = {"train_loss": [], "val_loss": [], "epoch_times": []}

    # early stopping state
    best_val     = float('inf')
    patience_ctr = 0
    stopped_epoch= EPOCHS

    print(f"\n{'─'*60}")
    print(f"  Training : {label}")
    print(f"  Params   : {count_params(model):,}")
    print(f"  Train N  : {n_train:,}   Val N: {n_val:,}")
    print(f"  Early stop: patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}")
    print(f"{'─'*60}")

    t_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        ep_t0 = time.perf_counter()

        # ── train ──
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                loss = criterion(model(batch), batch)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tr_loss += loss.item() * len(batch)

        # ── validate ──
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                    va_loss += criterion(model(batch), batch).item() * len(batch)

        tr_loss  /= n_train
        va_loss  /= n_val
        ep_time   = time.perf_counter() - ep_t0
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["epoch_times"].append(ep_time)

        # ── early stopping check ──
        improved = va_loss < (best_val - EARLY_STOP_MIN_DELTA)
        if improved:
            best_val     = va_loss
            patience_ctr = 0
        else:
            patience_ctr += 1

        gap    = va_loss - tr_loss
        status = "✓ improved" if improved else f"no improve ({patience_ctr}/{EARLY_STOP_PATIENCE})"
        if epoch % 5 == 0 or epoch == 1 or patience_ctr >= EARLY_STOP_PATIENCE:
            print(f"  Epoch {epoch:>3}/{EPOCHS}  "
                  f"train={tr_loss:.6f}  val={va_loss:.6f}  "
                  f"gap={gap:+.6f}  {ep_time:.1f}s  [{status}]")

        if patience_ctr >= EARLY_STOP_PATIENCE:
            stopped_epoch = epoch
            print(f"  ⚡ Early stopping at epoch {epoch} — val loss plateaued.")
            break

    train_time     = time.perf_counter() - t_start
    avg_epoch_time = np.mean(history["epoch_times"])
    print(f"  Stopped at epoch {stopped_epoch}/{EPOCHS} | "
          f"Total: {train_time:.1f}s | Avg/epoch: {avg_epoch_time:.1f}s | Best val: {best_val:.6f}")

    history["total_time"]     = train_time
    history["avg_epoch_time"] = avg_epoch_time
    history["stopped_epoch"]  = stopped_epoch
    history["best_val_loss"]  = best_val
    return history, train_time

def infer_scores(model, full_loader, is_flat=False, n_timing_runs=3):
    """
    Returns per-sample MSE scores and rigorous inference timing.
    - GPU warmup: 3 dummy batches before timing starts
    - Multiple timing runs on first batch for stable µs/sample estimate
    - Reports: mean inference µs/sample, std, throughput (samples/sec)
    """
    model.eval()

    # ── GPU warmup — prevents cold-start bias in timing ───────────────────────
    if DEVICE.type == "cuda":
        warm_batch = next(iter(full_loader)).to(DEVICE)
        with torch.no_grad():
            for _ in range(3):
                with torch.cuda.amp.autocast(enabled=True):
                    _ = model(warm_batch)
        torch.cuda.synchronize()

    # ── rigorous per-batch timing (n_timing_runs on first batch) ──────────────
    first_batch = next(iter(full_loader)).to(DEVICE)
    timing_runs = []
    with torch.no_grad():
        for _ in range(n_timing_runs):
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                _ = model(first_batch)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            timing_runs.append((time.perf_counter() - t0) / len(first_batch) * 1e6)

    infer_us_mean = float(np.mean(timing_runs))
    infer_us_std  = float(np.std(timing_runs))
    throughput    = int(1e6 / infer_us_mean)   # samples per second

    # ── full dataset scoring ───────────────────────────────────────────────────
    scores = []
    with torch.no_grad():
        for batch in full_loader:
            batch = batch.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                recon = model(batch)
            if is_flat:
                mse = ((batch - recon) ** 2).mean(dim=1)
            else:
                mse = ((batch - recon) ** 2).mean(dim=(1, 2))
            scores.append(mse.cpu().numpy())

    return np.concatenate(scores), infer_us_mean, infer_us_std, throughput

def detection_stats(scores):
    th        = np.percentile(scores, THRESHOLD_PERCENTILE)
    is_anom   = scores > th
    anom_rate = is_anom.mean() * 100
    sep_ratio = (scores[is_anom].mean() / scores[~is_anom].mean()
                 if (~is_anom).sum() > 0 else float('nan'))
    return {
        "mean"      : float(scores.mean()),
        "std"       : float(scores.std()),
        "min"       : float(scores.min()),
        "p25"       : float(np.percentile(scores, 25)),
        "p50"       : float(np.percentile(scores, 50)),
        "p75"       : float(np.percentile(scores, 75)),
        "p95"       : float(th),
        "p99"       : float(np.percentile(scores, 99)),
        "max"       : float(scores.max()),
        "anom_rate" : float(anom_rate),
        "sep_ratio" : float(sep_ratio),
    }

# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class VanillaAE(nn.Module):
    def __init__(self, d=18):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(16,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU(),nn.Linear(64,d))
    def forward(self, x): return self.dec(self.enc(x))

class SimpleLSTMAE(nn.Module):
    def __init__(self, d=18, h=128, layers=2):
        super().__init__()
        self.enc   = nn.LSTM(d, h, layers, batch_first=True)
        self.dec   = nn.LSTM(h, h, layers, batch_first=True)
        self.fc    = nn.Linear(h, d)
    def forward(self, x):
        _, (h, _) = self.enc(x)
        out, _    = self.dec(h[-1].unsqueeze(1).repeat(1, x.size(1), 1))
        return self.fc(out)

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__()
        self.pad  = (k-1)*d
        self.c1   = nn.Conv1d(in_ch, out_ch, k, padding=self.pad, dilation=d)
        self.c2   = nn.Conv1d(out_ch,out_ch, k, padding=self.pad, dilation=d)
        self.bn1  = nn.BatchNorm1d(out_ch)
        self.bn2  = nn.BatchNorm1d(out_ch)
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        o = self.c1(x); o = o[...,:-self.pad] if self.pad else o; o = F.relu(self.bn1(o))
        o = self.c2(o); o = o[...,:-self.pad] if self.pad else o; o = F.relu(self.bn2(o))
        return o + self.res(x)

class TCNAE(nn.Module):
    def __init__(self, d=18):
        super().__init__()
        self.enc = nn.Sequential(TCNBlock(d,64,d=1),TCNBlock(64,32,d=2),TCNBlock(32,16,d=4))
        self.dec = nn.Sequential(TCNBlock(16,32,d=4),TCNBlock(32,64,d=2),TCNBlock(64,d,d=1))
    def forward(self, x):
        return self.dec(self.enc(x.transpose(1,2))).transpose(1,2)

class TransformerAE(nn.Module):
    def __init__(self, d=18, dm=64, heads=2, layers=2, win=10):
        super().__init__()
        self.inp  = nn.Linear(d, dm)
        self.enc  = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dm,heads,128,0.1,batch_first=True), layers)
        self.dec  = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dm,heads,128,0.1,batch_first=True), layers)
        self.out  = nn.Linear(dm, d)
        self.pos  = nn.Embedding(win, dm)
        self.register_buffer('pidx', torch.arange(win))
    def forward(self, x):
        xp = self.inp(x) + self.pos(self.pidx)
        return self.out(self.dec(xp, self.enc(xp)))

class Attention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.w = nn.Linear(h*2, 1)
    def forward(self, x):
        return (torch.softmax(self.w(x), dim=1) * x).sum(dim=1)

class BidirLSTMAE(nn.Module):
    def __init__(self, d=18, h=128, layers=2):
        super().__init__()
        self.enc  = nn.LSTM(d, h, layers, batch_first=True, bidirectional=True)
        self.attn = Attention(h)
        self.proj = nn.Linear(h*2, h)
        self.dec  = nn.LSTM(h, h, layers, batch_first=True)
        self.fc   = nn.Linear(h, d)
    def forward(self, x):
        o, _  = self.enc(x)
        lat   = self.proj(self.attn(o)).unsqueeze(1).repeat(1, x.size(1), 1)
        out,_ = self.dec(lat)
        return self.fc(out)

# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════
records = []   # one entry per model

# ── 1. Vanilla AE ─────────────────────────────────────────────────────────────
flat_ds = FlatDataset(raw)
tr_l, va_l, full_l, ntr, nva = make_loaders(flat_ds, batch_train=4096)
m = VanillaAE().to(DEVICE)
hist, t_time = train_model(m, tr_l, va_l, ntr, nva, "Vanilla AE (Dense)")
scores, i_us, i_std, i_thr = infer_scores(m, full_l, is_flat=True)
np.save(os.path.join(OUT_DIR, "vanilla_ae_scores.npy"), scores)
records.append({
    "name": "Vanilla AE (Dense)", "arch": "Dense AE", "temporal": "No",
    "supervised": "No", "framework": "PyTorch", "params": count_params(m),
    "train_s": t_time, "infer_us": i_us, "infer_std": i_std, "throughput": i_thr,
    "final_train_loss" : hist["train_loss"][-1],
    "final_val_loss"   : hist["val_loss"][-1],
    "overfit_gap"      : hist["val_loss"][-1] - hist["train_loss"][-1],
    "avg_epoch_time"   : hist["avg_epoch_time"],
    "stats": detection_stats(scores), "history": hist,
})

# ── 2. Simple LSTM-AE ─────────────────────────────────────────────────────────
win_ds = WindowDataset(raw, WINDOW)
tr_l, va_l, full_l, ntr, nva = make_loaders(win_ds, batch_train=1024)
m = SimpleLSTMAE().to(DEVICE)
hist, t_time = train_model(m, tr_l, va_l, ntr, nva, "Simple LSTM-AE (Malhotra 2015)")
scores, i_us, i_std, i_thr = infer_scores(m, full_l)
np.save(os.path.join(OUT_DIR, "lstm_ae_simple_scores.npy"), scores)
records.append({
    "name": "Simple LSTM-AE", "arch": "LSTM AE", "temporal": "Yes",
    "supervised": "No", "framework": "PyTorch", "params": count_params(m),
    "train_s": t_time, "infer_us": i_us, "infer_std": i_std, "throughput": i_thr,
    "final_train_loss" : hist["train_loss"][-1],
    "final_val_loss"   : hist["val_loss"][-1],
    "overfit_gap"      : hist["val_loss"][-1] - hist["train_loss"][-1],
    "avg_epoch_time"   : hist["avg_epoch_time"],
    "stats": detection_stats(scores), "history": hist,
})

# ── 3. TCN-AE ─────────────────────────────────────────────────────────────────
tr_l, va_l, full_l, ntr, nva = make_loaders(win_ds, batch_train=1024)
m = TCNAE().to(DEVICE)
hist, t_time = train_model(m, tr_l, va_l, ntr, nva, "TCN Autoencoder (Springer 2024)")
scores, i_us, i_std, i_thr = infer_scores(m, full_l)
np.save(os.path.join(OUT_DIR, "tcn_ae_scores.npy"), scores)
records.append({
    "name": "TCN Autoencoder", "arch": "Conv1D AE", "temporal": "Yes",
    "supervised": "No", "framework": "PyTorch", "params": count_params(m),
    "train_s": t_time, "infer_us": i_us, "infer_std": i_std, "throughput": i_thr,
    "final_train_loss" : hist["train_loss"][-1],
    "final_val_loss"   : hist["val_loss"][-1],
    "overfit_gap"      : hist["val_loss"][-1] - hist["train_loss"][-1],
    "avg_epoch_time"   : hist["avg_epoch_time"],
    "stats": detection_stats(scores), "history": hist,
})

# ── 4. Transformer-AE ─────────────────────────────────────────────────────────
tr_l, va_l, full_l, ntr, nva = make_loaders(win_ds, batch_train=512)
m = TransformerAE(win=WINDOW).to(DEVICE)
hist, t_time = train_model(m, tr_l, va_l, ntr, nva, "Transformer AE (MDPI 2024)")
scores, i_us, i_std, i_thr = infer_scores(m, full_l)
np.save(os.path.join(OUT_DIR, "transformer_ae_scores.npy"), scores)
records.append({
    "name": "Transformer AE", "arch": "Transformer AE", "temporal": "Yes",
    "supervised": "No", "framework": "PyTorch", "params": count_params(m),
    "train_s": t_time, "infer_us": i_us, "infer_std": i_std, "throughput": i_thr,
    "final_train_loss" : hist["train_loss"][-1],
    "final_val_loss"   : hist["val_loss"][-1],
    "overfit_gap"      : hist["val_loss"][-1] - hist["train_loss"][-1],
    "avg_epoch_time"   : hist["avg_epoch_time"],
    "stats": detection_stats(scores), "history": hist,
})

# ── 5. Bidir LSTM-AE + Attention (Proposed) ───────────────────────────────────
tr_l, va_l, full_l, ntr, nva = make_loaders(win_ds, batch_train=1024)
m = BidirLSTMAE().to(DEVICE)
hist, t_time = train_model(m, tr_l, va_l, ntr, nva, "Bidir LSTM-AE + Attention (Proposed)")
scores, i_us, i_std, i_thr = infer_scores(m, full_l)
np.save(os.path.join(OUT_DIR, "bidir_lstm_ae_scores.npy"), scores)
records.append({
    "name": "Bidir LSTM-AE+Attn (Ours)", "arch": "Bidir LSTM AE", "temporal": "Yes",
    "supervised": "No", "framework": "PyTorch", "params": count_params(m),
    "train_s": t_time, "infer_us": i_us, "infer_std": i_std, "throughput": i_thr,
    "final_train_loss" : hist["train_loss"][-1],
    "final_val_loss"   : hist["val_loss"][-1],
    "overfit_gap"      : hist["val_loss"][-1] - hist["train_loss"][-1],
    "avg_epoch_time"   : hist["avg_epoch_time"],
    "stats": detection_stats(scores), "history": hist,
})

# ── 6. Isolation Forest ───────────────────────────────────────────────────────
print("\n─────────────────────────────────────────────────────")
print("  Running: Isolation Forest")
t0 = time.perf_counter()
IF = IsolationForest(n_estimators=100, contamination=0.05, random_state=SEED, n_jobs=-1)
IF.fit(raw)
t_time = time.perf_counter() - t0
# warmup + timing for IF
_ = IF.score_samples(raw[:1000])
runs = []
for _ in range(3):
    t0 = time.perf_counter()
    if_scores = -IF.score_samples(raw)
    runs.append((time.perf_counter() - t0) / len(raw) * 1e6)
i_us  = float(np.mean(runs))
i_std = float(np.std(runs))
i_thr = int(1e6 / i_us)
np.save(os.path.join(OUT_DIR, "isolation_forest_scores.npy"), if_scores)
records.append({
    "name": "Isolation Forest", "arch": "Tree Ensemble", "temporal": "No",
    "supervised": "No", "framework": "sklearn", "params": "N/A",
    "train_s": t_time, "infer_us": i_us, "infer_std": i_std, "throughput": i_thr, "avg_epoch_time": "N/A",
    "final_train_loss": "N/A", "final_val_loss": "N/A", "overfit_gap": "N/A",
    "stats": detection_stats(if_scores), "history": None,
})
print(f"  Done in {t_time:.1f}s")

# ── 7. One-Class SVM ──────────────────────────────────────────────────────────
print("\n─────────────────────────────────────────────────────")
print("  Running: One-Class SVM (fit on 20k subsample)")
fit_idx = np.random.default_rng(SEED).choice(len(raw), 20000, replace=False)
t0 = time.perf_counter()
svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
svm.fit(raw[fit_idx])
t_time = time.perf_counter() - t0
# warmup + timing for OCSVM
_ = svm.score_samples(raw[:1000])
runs = []
for _ in range(3):
    t0 = time.perf_counter()
    svm_scores = -svm.score_samples(raw)
    runs.append((time.perf_counter() - t0) / len(raw) * 1e6)
i_us  = float(np.mean(runs))
i_std = float(np.std(runs))
i_thr = int(1e6 / i_us)
np.save(os.path.join(OUT_DIR, "ocsvm_scores.npy"), svm_scores)
records.append({
    "name": "One-Class SVM", "arch": "Kernel Boundary", "temporal": "No",
    "supervised": "No", "framework": "sklearn", "params": "N/A",
    "train_s": t_time, "infer_us": i_us, "infer_std": i_std, "throughput": i_thr, "avg_epoch_time": "N/A",
    "final_train_loss": "N/A", "final_val_loss": "N/A", "overfit_gap": "N/A",
    "stats": detection_stats(svm_scores), "history": None,
})
print(f"  Done in {t_time:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# PRINT PUBLICATION-READY TABLES
# ══════════════════════════════════════════════════════════════════════════════
W = 140
print(f"\n\n{'═'*W}")
print("  COMPREHENSIVE MODEL COMPARISON")
print("  Dataset  : CESNET Time-Series 24  |  Aggregation: 10-minute")
print("  Subset   : 3,000,000 rows  |  Split: 80% train / 20% val  |  Seed: 42")
print("  Preproc  : log1p + MinMaxScaler  |  Threshold: 95th percentile MSE")
print(f"{'═'*W}")

# ── TABLE I: Architecture & Complexity ────────────────────────────────────────
print("\nTABLE I — Architecture & Computational Complexity")
print(f"{'─'*W}")
hdr = (f"{'Model':<28} {'Architecture':<18} {'Temporal':<10} {'Params':>9} "
       f"{'Train(s)':>10} {'Infer µs':>10} {'±std':>7} {'Throughput/s':>14}")
print(hdr)
print(f"{'─'*W}")
for r in records:
    p   = f"{r['params']:,}" if isinstance(r['params'], int) else r['params']
    tr  = f"{r['train_s']:.1f}"   if isinstance(r['train_s'],  float) else r['train_s']
    iu  = f"{r['infer_us']:.3f}"  if isinstance(r['infer_us'], float) else r['infer_us']
    ist = f"{r['infer_std']:.3f}" if isinstance(r.get('infer_std'), float) else "N/A"
    thr = f"{r['throughput']:,}"  if isinstance(r.get('throughput'), int)  else "N/A"
    print(f"{r['name']:<28} {r['arch']:<18} {r['temporal']:<10} {p:>9} "
          f"{tr:>10} {iu:>10} {ist:>7} {thr:>14}")

# ── TABLE II: Convergence ─────────────────────────────────────────────────────
print(f"\n\nTABLE II — Convergence Behaviour")
print(f"{'─'*W}")
hdr2 = (f"{'Model':<28} {'Stopped Ep':>11} {'Best Val Loss':>15} {'Final Train':>13} "
        f"{'Overfit Gap':>13}  {'Assessment'}")
print(hdr2)
print(f"{'─'*W}")
for r in records:
    if r['final_train_loss'] == "N/A":
        print(f"{r['name']:<28} {'N/A':>11} {'N/A':>15} {'N/A':>13} {'N/A':>13}  N/A (no gradient training)")
        continue
    gap        = r['overfit_gap']
    stopped    = r['history']['stopped_epoch']
    best_val   = r['history']['best_val_loss']
    assessment = ("Good generalisation" if abs(gap) < 0.001
                  else "Slight overfit" if gap > 0
                  else "Slight underfit")
    early      = " (early stop)" if stopped < EPOCHS else ""
    print(f"{r['name']:<28} {str(stopped)+early:>11} {best_val:>15.6f} {r['final_train_loss']:>13.6f} "
          f"{gap:>+13.6f}  {assessment}")

# ── TABLE III: Detection Performance ─────────────────────────────────────────
print(f"\n\nTABLE III — Anomaly Detection Performance")
print(f"{'─'*W}")
hdr3 = (f"{'Model':<28} {'Mean':>10} {'Std':>10} {'Median':>9} "
        f"{'p95(thr)':>10} {'p99':>10} {'Anom%':>8} {'Sep.Ratio':>11}")
print(hdr3)
print(f"{'─'*W}")
for r in records:
    s = r['stats']
    print(f"{r['name']:<28} {s['mean']:>10.6f} {s['std']:>10.6f} {s['p50']:>9.6f} "
          f"{s['p95']:>10.6f} {s['p99']:>10.6f} {s['anom_rate']:>8.2f} {s['sep_ratio']:>11.3f}x")

# ── TABLE IV: Score Distribution Quartiles ────────────────────────────────────
print(f"\n\nTABLE IV — Score Distribution Quartiles")
print(f"{'─'*W}")
hdr4 = f"{'Model':<28} {'Min':>12} {'Q1(p25)':>12} {'Median':>12} {'Q3(p75)':>12} {'Max':>12}"
print(hdr4)
print(f"{'─'*W}")
for r in records:
    s = r['stats']
    print(f"{r['name']:<28} {s['min']:>12.6f} {s['p25']:>12.6f} {s['p50']:>12.6f} "
          f"{s['p75']:>12.6f} {s['max']:>12.6f}")

# ── TABLE V: Training Efficiency ──────────────────────────────────────────────
print(f"\n\nTABLE V — Training Efficiency (Primary Advantage of Proposed Model)")
print(f"{'─'*W}")
hdr5 = (f"{'Model':<28} {'Params':>9} {'Total Train(s)':>16} {'Avg/Epoch(s)':>14} "
        f"{'Infer(µs)':>11} {'Params/Sep.Ratio':>17}  {'Efficiency Rank'}")
print(hdr5)
print(f"{'─'*W}")

# rank by total training time (lower = better), only deep models
deep = [r for r in records if isinstance(r['train_s'], float)]
deep_sorted = sorted(deep, key=lambda r: r['train_s'])
rank_map = {r['name']: i+1 for i, r in enumerate(deep_sorted)}

for r in records:
    p   = f"{r['params']:,}" if isinstance(r['params'], int) else r['params']
    tr  = f"{r['train_s']:.1f}"       if isinstance(r['train_s'], float)       else r['train_s']
    ep  = f"{r['avg_epoch_time']:.1f}" if isinstance(r['avg_epoch_time'], float) else r['avg_epoch_time']
    it  = f"{r['infer_us']:.3f}"      if isinstance(r['infer_us'], float)       else r['infer_us']
    sep = r['stats']['sep_ratio']
    par = r['params']
    # efficiency score = sep_ratio / (params/1000) — more separation per 1k params
    if isinstance(par, int) and sep > 0:
        eff = f"{sep / (par/1000):.4f}"
    else:
        eff = "N/A"
    rank = f"#{rank_map[r['name']]}" if r['name'] in rank_map else "N/A"
    print(f"{r['name']:<28} {p:>9} {tr:>16} {ep:>14} {it:>11} {eff:>17}  {rank}")

print(f"\n  * Efficiency Score = Sep.Ratio / (Params / 1000)  "
      f"— separability achieved per 1,000 parameters")
print(f"  * Training Rank: #1 = fastest to train among deep learning models")

print(f"\n{'═'*W}")
print("LEGEND:")
print("  Sep.Ratio      = mean(anomaly scores) / mean(normal scores)  — higher = better separation")
print("  Overfit Gap    = val_loss − train_loss  — near 0 is ideal")
print("  Infer(µs)      = microseconds per sample during inference (lower = faster deployment)")
print("  ±std           = timing stability across 3 runs (lower = more consistent)")
print("  Throughput/s   = samples processed per second — key for real-time ISP deployment")
print("  Avg/Epoch(s)   = average wall-clock time per training epoch")
print("  Efficiency     = Sep.Ratio / (Params/1000) — detection quality per 1k parameters")
print("  Training Rank  = #1 fastest total training time among deep learning models")
print("  IF / OCSVM     = no gradient training; fit time shown instead of epoch time")
print("  All .npy score files saved to:", OUT_DIR)
print(f"{'═'*W}\n")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

COLORS = {
    "Vanilla AE (Dense)"         : "#7f8c8d",
    "Simple LSTM-AE"             : "#3498db",
    "TCN Autoencoder"            : "#e67e22",
    "Transformer AE"             : "#9b59b6",
    "Bidir LSTM-AE+Attn (Ours)"  : "#e74c3c",
    "Isolation Forest"           : "#1abc9c",
    "One-Class SVM"              : "#f39c12",
}
PROPOSED = "Bidir LSTM-AE+Attn (Ours)"
deep_records = [r for r in records if r['history'] is not None]

print("Generating plots...")

# ── PLOT 1: Train & Validation Loss Curves (all 5 deep models) ────────────────
fig, axes = plt.subplots(1, len(deep_records), figsize=(5*len(deep_records), 4), sharey=False)
fig.suptitle("Training & Validation Loss per Epoch\n(Lower = Better Reconstruction of Normal Traffic)",
             fontsize=13, fontweight='bold', y=1.02)
for ax, r in zip(axes, deep_records):
    epochs = range(1, EPOCHS + 1)
    c = COLORS[r['name']]
    ax.plot(epochs, r['history']['train_loss'], color=c,      lw=2, label='Train Loss')
    ax.plot(epochs, r['history']['val_loss'],   color=c, lw=2, ls='--', label='Val Loss')
    ax.fill_between(epochs, r['history']['train_loss'], r['history']['val_loss'],
                    alpha=0.12, color=c)
    ax.set_title(r['name'], fontsize=9, fontweight='bold',
                 color='#e74c3c' if r['name'] == PROPOSED else 'black')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # annotate final val loss
    ax.annotate(f"val={r['history']['val_loss'][-1]:.5f}",
                xy=(EPOCHS, r['history']['val_loss'][-1]),
                xytext=(-30, 8), textcoords='offset points',
                fontsize=7, color=c,
                arrowprops=dict(arrowstyle='->', color=c, lw=1))
plt.tight_layout()
p = os.path.join(PLOT_DIR, "plot1_loss_curves.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")

# ── PLOT 2: Training Time Comparison (bar) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
names  = [r['name'] for r in records]
times  = [r['train_s'] if isinstance(r['train_s'], float) else 0 for r in records]
colors = [COLORS.get(n, '#95a5a6') for n in names]
bars   = ax.barh(names, times, color=colors, edgecolor='white', height=0.6)
for bar, t in zip(bars, times):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"{t:.1f}s", va='center', fontsize=9)
# highlight proposed
proposed_idx = names.index(PROPOSED)
bars[proposed_idx].set_edgecolor('#c0392b')
bars[proposed_idx].set_linewidth(2.5)
ax.set_xlabel("Total Training Time (seconds)", fontsize=11)
ax.set_title("Model Training Time Comparison\n(★ Proposed Model highlighted in red border)",
             fontsize=12, fontweight='bold')
ax.axvline(times[proposed_idx], color='#e74c3c', ls='--', lw=1.2, alpha=0.5)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
p = os.path.join(PLOT_DIR, "plot2_training_time.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")

# ── PLOT 3: Epoch-wise Training Time (avg seconds/epoch) ──────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ep_names  = [r['name'] for r in deep_records]
ep_times  = [r['avg_epoch_time'] for r in deep_records]
ep_colors = [COLORS.get(n, '#95a5a6') for n in ep_names]
bars = ax.bar(ep_names, ep_times, color=ep_colors, edgecolor='white', width=0.6)
for bar, t in zip(bars, ep_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{t:.1f}s", ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel("Avg Time per Epoch (seconds)", fontsize=11)
ax.set_title("Average Training Time per Epoch\n(Lower = More Efficient Model)",
             fontsize=12, fontweight='bold')
ax.set_xticks(range(len(ep_names)))
ax.set_xticklabels(ep_names, rotation=20, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
# star on proposed
if PROPOSED in ep_names:
    pi = ep_names.index(PROPOSED)
    bars[pi].set_edgecolor('#c0392b')
    bars[pi].set_linewidth(2.5)
plt.tight_layout()
p = os.path.join(PLOT_DIR, "plot3_epoch_time.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")

# ── PLOT 3b: Inference Time with Error Bars ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

inf_names  = [r['name'] for r in records]
inf_us     = [r['infer_us'] if isinstance(r['infer_us'], float) else 0 for r in records]
inf_std    = [r.get('infer_std', 0) if isinstance(r.get('infer_std'), float) else 0 for r in records]
inf_thr    = [r.get('throughput', 0) if isinstance(r.get('throughput'), int) else 0 for r in records]
inf_colors = [COLORS.get(n, '#95a5a6') for n in inf_names]

# left: µs per sample with error bars
ax = axes[0]
bars = ax.barh(inf_names, inf_us, xerr=inf_std, color=inf_colors,
               edgecolor='white', height=0.6,
               error_kw=dict(ecolor='#2c3e50', capsize=4, lw=1.5))
for bar, v, s in zip(bars, inf_us, inf_std):
    ax.text(bar.get_width() + s + 0.002,
            bar.get_y() + bar.get_height()/2,
            f"{v:.3f}µs", va='center', fontsize=8)
if PROPOSED in inf_names:
    pi = inf_names.index(PROPOSED)
    bars[pi].set_edgecolor('#c0392b')
    bars[pi].set_linewidth(2.5)
ax.set_xlabel("Inference Time per Sample (µs)  — Lower is Better", fontsize=10)
ax.set_title("Inference Latency Comparison\n(±std over 3 timed runs, GPU warmed up)",
             fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

# right: throughput (samples/sec)
ax = axes[1]
bars2 = ax.barh(inf_names, inf_thr, color=inf_colors, edgecolor='white', height=0.6)
for bar, v in zip(bars2, inf_thr):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
            f"{v:,}/s", va='center', fontsize=8)
if PROPOSED in inf_names:
    bars2[inf_names.index(PROPOSED)].set_edgecolor('#c0392b')
    bars2[inf_names.index(PROPOSED)].set_linewidth(2.5)
ax.set_xlabel("Throughput (samples/second)  — Higher is Better", fontsize=10)
ax.set_title("Inference Throughput Comparison\n(Real-time ISP deployment suitability)",
             fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
p = os.path.join(PLOT_DIR, "plot3b_inference_time.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")

# ── PLOT 4: Anomaly Score Distribution (KDE-style histogram) ──────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
all_score_files = {
    "Vanilla AE (Dense)"        : "vanilla_ae_scores.npy",
    "Simple LSTM-AE"            : "lstm_ae_simple_scores.npy",
    "TCN Autoencoder"           : "tcn_ae_scores.npy",
    "Transformer AE"            : "transformer_ae_scores.npy",
    "Bidir LSTM-AE+Attn (Ours)" : "bidir_lstm_ae_scores.npy",
    "Isolation Forest"          : "isolation_forest_scores.npy",
    "One-Class SVM"             : "ocsvm_scores.npy",
}
for ax, (name, fname) in zip(axes, all_score_files.items()):
    fpath = os.path.join(OUT_DIR, fname)
    if not os.path.exists(fpath):
        ax.text(0.5, 0.5, "Not run yet", ha='center', va='center')
        ax.set_title(name, fontsize=8)
        continue
    sc  = np.load(fpath)
    th  = np.percentile(sc, THRESHOLD_PERCENTILE)
    c   = COLORS.get(name, '#95a5a6')
    # clip for readability
    sc_clip = np.clip(sc, 0, np.percentile(sc, 99.5))
    ax.hist(sc_clip[sc_clip <= th], bins=80, color=c,      alpha=0.7, label='Normal')
    ax.hist(sc_clip[sc_clip >  th], bins=40, color='#e74c3c', alpha=0.8, label='Anomaly')
    ax.axvline(th, color='black', ls='--', lw=1.5, label=f'p95={th:.5f}')
    ax.set_title(name, fontsize=9,
                 fontweight='bold' if name == PROPOSED else 'normal',
                 color='#e74c3c' if name == PROPOSED else 'black')
    ax.set_xlabel("Anomaly Score (MSE)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
axes[-1].set_visible(False)
fig.suptitle("Anomaly Score Distributions per Model\n(Red = detected anomalies, dashed = p95 threshold)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
p = os.path.join(PLOT_DIR, "plot4_score_distributions.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")

# ── PLOT 5: Separability Ratio Comparison ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sep_names  = [r['name'] for r in records]
sep_vals   = [r['stats']['sep_ratio'] for r in records]
sep_colors = [COLORS.get(n, '#95a5a6') for n in sep_names]
bars = ax.barh(sep_names, sep_vals, color=sep_colors, edgecolor='white', height=0.6)
for bar, v in zip(bars, sep_vals):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{v:.3f}x", va='center', fontsize=9)
if PROPOSED in sep_names:
    pi = sep_names.index(PROPOSED)
    bars[pi].set_edgecolor('#c0392b')
    bars[pi].set_linewidth(2.5)
ax.set_xlabel("Separability Ratio  (Anomaly Mean Score / Normal Mean Score)", fontsize=10)
ax.set_title("Anomaly Score Separability Ratio per Model\n"
             "(Higher = Clearer boundary between normal and anomalous traffic)",
             fontsize=12, fontweight='bold')
ax.axvline(1.0, color='gray', ls=':', lw=1)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
p = os.path.join(PLOT_DIR, "plot5_separability.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")

# ── PLOT 6: Params vs Separability (Efficiency Frontier) ──────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for r in deep_records:
    x = r['params']
    y = r['stats']['sep_ratio']
    c = COLORS.get(r['name'], '#95a5a6')
    is_proposed = r['name'] == PROPOSED
    ax.scatter(x, y, s=220 if is_proposed else 120, color=c,
               zorder=5, edgecolors='#c0392b' if is_proposed else 'white',
               linewidths=2.5 if is_proposed else 1)
    ax.annotate(r['name'], (x, y),
                textcoords='offset points',
                xytext=(8, 4 if not is_proposed else 10),
                fontsize=8,
                fontweight='bold' if is_proposed else 'normal',
                color='#c0392b' if is_proposed else 'black')
ax.set_xlabel("Number of Trainable Parameters", fontsize=11)
ax.set_ylabel("Separability Ratio", fontsize=11)
ax.set_title("Efficiency Frontier: Parameters vs Detection Separability\n"
             "(Top-left = best: fewer params, higher separability)",
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
# draw ideal region annotation
ax.annotate("← Fewer params\nBetter →", xy=(0.05, 0.92),
            xycoords='axes fraction', fontsize=9, color='green', alpha=0.7)
plt.tight_layout()
p = os.path.join(PLOT_DIR, "plot6_efficiency_frontier.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")

# ── PLOT 7: Convergence Speed (normalised loss) ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for r in deep_records:
    losses = np.array(r['history']['val_loss'])
    norm   = (losses - losses.min()) / (losses.max() - losses.min() + 1e-12)
    c      = COLORS.get(r['name'], '#95a5a6')
    lw     = 3 if r['name'] == PROPOSED else 1.5
    ls     = '-' if r['name'] == PROPOSED else '--'
    ax.plot(range(1, EPOCHS+1), norm, color=c, lw=lw, ls=ls, label=r['name'])
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Normalised Validation Loss  (0=best, 1=worst)", fontsize=10)
ax.set_title("Convergence Speed Comparison\n"
             "(Steeper drop = faster convergence, lower final = better fit)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()
p = os.path.join(PLOT_DIR, "plot7_convergence_speed.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")

# ── PLOT 8: Box Plot of Anomaly Scores ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
box_data, box_labels, box_colors = [], [], []
for name, fname in all_score_files.items():
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        sc = np.load(fpath)
        box_data.append(np.clip(sc, 0, np.percentile(sc, 99)))
        box_labels.append(name)
        box_colors.append(COLORS.get(name, '#95a5a6'))
bp = ax.boxplot(box_data, patch_artist=True, vert=True,
                medianprops=dict(color='black', lw=2),
                whiskerprops=dict(lw=1.2),
                flierprops=dict(marker='.', ms=2, alpha=0.3))
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_xticks(range(1, len(box_labels)+1))
ax.set_xticklabels(box_labels, rotation=25, ha='right', fontsize=9)
ax.set_ylabel("Anomaly Score (clipped at p99)", fontsize=11)
ax.set_title("Anomaly Score Box Plot per Model\n"
             "(Tighter box with higher median anomaly range = better separation)",
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
p = os.path.join(PLOT_DIR, "plot8_boxplot_scores.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")

print(f"\nAll plots saved to: {PLOT_DIR}")
print("Plots generated:")
print("  plot1_loss_curves.png        — Train & Val loss per epoch (all 5 deep models)")
print("  plot2_training_time.png      — Total training time bar chart")
print("  plot3_epoch_time.png         — Avg time per epoch (efficiency)")
print("  plot3b_inference_time.png    — Inference µs/sample + throughput with error bars")
print("  plot4_score_distributions.png— Anomaly score histogram per model")
print("  plot5_separability.png       — Separability ratio comparison")
print("  plot6_efficiency_frontier.png— Params vs Separability scatter (efficiency claim)")
print("  plot7_convergence_speed.png  — Normalised val loss convergence curves")
print("  plot8_boxplot_scores.png     — Score distribution box plots")
