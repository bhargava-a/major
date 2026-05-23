"""
Model 5b: Proposed Model — Bidirectional LSTM Autoencoder + Attention
Fast-path version: entire dataset on GPU, manual batching, no disk I/O.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
NPZ_PATH   = r"c:\Users\bharg\OneDrive\Documents\GPU_package\data\cesnet_10min.npz"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW     = 10
BATCH_SIZE = 256           # halves step count vs 128
EPOCHS     = 15            # trimmed — cosine + patience will stop earlier
LR         = 2e-3          # linear scale for 2x batch — keeps effective step size
SUBSAMPLE  = 3_000_000     # matches baselines — essential for fair comparison
VAL_SPLIT  = 0.20
SEED       = 42
GRAD_CLIP  = 1.0
DROPOUT    = 0.3
HIDDEN     = 128
N_LAYERS   = 2
THRESHOLD_PERCENTILE = 95
EARLY_STOP_PATIENCE  = 6   # give cosine LR room to work
EARLY_STOP_MIN_DELTA = 1e-8
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

print(f"Device : {DEVICE}")
print(f"TF32   : enabled  |  cudnn.benchmark: enabled\n")

# ── Load data directly to GPU (no DataLoader, no per-batch transfer) ──────────
raw = np.load(NPZ_PATH)['data']
idx = np.random.default_rng(SEED).choice(len(raw), SUBSAMPLE, replace=False)
idx.sort()
data_gpu = torch.tensor(raw[idx].astype(np.float32), device=DEVICE)   # (N, F)

N_total = data_gpu.shape[0] - WINDOW + 1
n_val   = int(N_total * VAL_SPLIT)
n_train = N_total - n_val

# Shuffle window-start indices deterministically, split into train/val
g = torch.Generator(device='cpu').manual_seed(SEED)
perm = torch.randperm(N_total, generator=g)
train_starts = perm[:n_train].to(DEVICE)
val_starts   = perm[n_train:].to(DEVICE)
all_starts   = torch.arange(N_total, device=DEVICE)

print(f"Total windows : {N_total:,}")
print(f"Train / Val   : {n_train:,} / {n_val:,}")
print(f"Dataset on GPU: {data_gpu.numel() * 4 / 1e6:.1f} MB\n")

def gather_windows(data, starts, window):
    """Build (B, T, F) batch from start indices using advanced indexing."""
    # starts: (B,)  → offsets (B, T) = starts[:, None] + arange(T)
    offsets = starts.unsqueeze(1) + torch.arange(window, device=data.device)
    return data[offsets]   # (B, T, F)

# ── Model ─────────────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, enc_out):
        w = F.softmax(self.attention(enc_out), dim=1)
        return torch.sum(w * enc_out, dim=1), w

class BidirLSTMAE(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=HIDDEN,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        self.attention   = Attention(hidden_dim * 2)
        self.fc_compress = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm  = nn.LayerNorm(hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, n_layers,
                               batch_first=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, input_dim)
        )
    def forward(self, x):
        enc_out, _ = self.encoder(x)
        ctx, _     = self.attention(enc_out)
        latent     = self.layer_norm(F.relu(self.fc_compress(ctx)))
        dec_in = latent.unsqueeze(1).repeat(1, x.size(1), 1)
        h_dec  = latent.unsqueeze(0).repeat(self.n_layers, 1, 1)
        c_dec  = torch.zeros_like(h_dec)
        dec_out, _ = self.decoder(dec_in, (h_dec, c_dec))
        return self.output_layer(dec_out)

input_dim = data_gpu.shape[1]
model     = BidirLSTMAE(input_dim=input_dim).to(DEVICE)
n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model params  : {n_params:,}")
print(f"Architecture  : BiDir LSTM encoder (h={HIDDEN}, layers={N_LAYERS})")
print(f"               2-layer attention | latent-seeded decoder | 2-layer output\n")

# ── Training ──────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

best_val      = float('inf')
best_state    = None           # in-memory checkpoint, no disk I/O
patience_ctr  = 0
stopped_ep    = EPOCHS
history       = {"train": [], "val": [], "epoch_t": []}

print(f"{'─'*65}")
print(f"  Training: Bidir LSTM-AE + Attention")
print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR} | Early stop: {EARLY_STOP_PATIENCE}")
print(f"{'─'*65}")

t_start = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    ep_t0 = time.perf_counter()

    # ── train ─────────────────────────────────────────────────────────────────
    model.train()
    perm_ep = train_starts[torch.randperm(n_train, device=DEVICE)]
    tr_loss = 0.0
    n_seen  = 0
    for i in range(0, n_train, BATCH_SIZE):
        batch_starts = perm_ep[i:i + BATCH_SIZE]
        batch        = gather_windows(data_gpu, batch_starts, WINDOW)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            loss = criterion(model(batch), batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        tr_loss += loss.item() * batch.size(0)
        n_seen  += batch.size(0)

    # ── validate ──────────────────────────────────────────────────────────────
    model.eval()
    va_loss = 0.0
    v_seen  = 0
    with torch.no_grad():
        for i in range(0, n_val, 4096):
            batch_starts = val_starts[i:i + 4096]
            batch        = gather_windows(data_gpu, batch_starts, WINDOW)
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                v = criterion(model(batch), batch)
            va_loss += v.item() * batch.size(0)
            v_seen  += batch.size(0)

    tr_loss /= n_seen
    va_loss /= v_seen
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    ep_time = time.perf_counter() - ep_t0

    scheduler.step()   # CosineAnnealingLR steps per-epoch, no val arg
    cur_lr = optimizer.param_groups[0]['lr']
    history["train"].append(tr_loss)
    history["val"].append(va_loss)
    history["epoch_t"].append(ep_time)

    improved = va_loss < (best_val - EARLY_STOP_MIN_DELTA)
    if improved:
        best_val = va_loss
        patience_ctr = 0
        # in-memory copy — no disk write
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        patience_ctr += 1

    status = "OK" if improved else f"x ({patience_ctr}/{EARLY_STOP_PATIENCE})"
    print(f"  Epoch {epoch:>3}/{EPOCHS}  "
          f"train={tr_loss:.6f}  val={va_loss:.6f}  "
          f"gap={va_loss-tr_loss:+.6f}  lr={cur_lr:.2e}  {ep_time:.1f}s  [{status}]")

    if patience_ctr >= EARLY_STOP_PATIENCE:
        stopped_ep = epoch
        print(f"  Early stopping at epoch {epoch}")
        break

train_time = time.perf_counter() - t_start
avg_epoch  = float(np.mean(history["epoch_t"]))
print(f"\n  Total    : {train_time:.1f}s")
print(f"  Avg/epoch: {avg_epoch:.1f}s")
print(f"  Best val : {best_val:.6f}  at epoch {stopped_ep}")

# ── Restore best weights ──────────────────────────────────────────────────────
if best_state is not None:
    model.load_state_dict(best_state)
model.eval()

# ── Inference timing ──────────────────────────────────────────────────────────
first_batch = gather_windows(data_gpu, all_starts[:4096], WINDOW)
if DEVICE.type == "cuda":
    with torch.no_grad():
        for _ in range(5):
            _ = model(first_batch)
    torch.cuda.synchronize()

timing_runs = []
for _ in range(5):
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(first_batch)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    timing_runs.append((time.perf_counter() - t0) / first_batch.size(0) * 1e6)

infer_us   = float(np.mean(timing_runs))
infer_std  = float(np.std(timing_runs))
throughput = int(1e6 / infer_us)

# ── Full dataset anomaly scores ───────────────────────────────────────────────
score_chunks = []
with torch.no_grad():
    for i in range(0, N_total, 4096):
        batch_starts = all_starts[i:i + 4096]
        batch        = gather_windows(data_gpu, batch_starts, WINDOW)
        recon        = model(batch)
        mse          = ((batch - recon) ** 2).mean(dim=(1, 2))
        score_chunks.append(mse.cpu().numpy())

scores    = np.concatenate(score_chunks)
threshold = np.percentile(scores, THRESHOLD_PERCENTILE)
anom_rate = (scores > threshold).sum() / len(scores) * 100
sep_ratio = scores[scores > threshold].mean() / scores[scores <= threshold].mean()

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  BIDIR LSTM-AE + ATTENTION — RESULTS")
print(f"{'='*60}")
print(f"  Params          : {n_params:,}")
print(f"  Train time      : {train_time:.1f}s  (stopped ep {stopped_ep}/{EPOCHS})")
print(f"  Avg/epoch       : {avg_epoch:.1f}s")
print(f"  Best val loss   : {best_val:.6f}")
print(f"  Final train loss: {history['train'][-1]:.6f}")
print(f"  Overfit gap     : {history['val'][-1] - history['train'][-1]:+.6f}")
print(f"{'-'*60}")
print(f"  Mean MSE        : {scores.mean():.6f}")
print(f"  Threshold (p95) : {threshold:.6f}")
print(f"  Anomaly rate    : {anom_rate:.2f}%")
print(f"  Sep. ratio      : {sep_ratio:.3f}x")
print(f"{'-'*60}")
print(f"  Infer us/sample : {infer_us:.3f} +/- {infer_std:.3f}")
print(f"  Throughput      : {throughput:,} samples/sec")
print(f"{'='*60}")
