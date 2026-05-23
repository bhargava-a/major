"""
Model 1: Vanilla Autoencoder (Dense AE)
Reference: Hinton & Salakhutdinov (2006) — applied to anomaly detection
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time, os

NPZ_PATH   = r"c:\Users\bharg\OneDrive\Documents\GPU_package\data\cesnet_10min.npz"
OUT_DIR    = r"c:\Users\bharg\OneDrive\Documents\GPU_package\results"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4096        # dense model is tiny — push GPU hard
EPOCHS     = 20
LR         = 1e-3
SUBSAMPLE  = 3_000_000   # 3M rows from 18M — representative and fast
THRESHOLD_PERCENTILE = 95

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# Load + subsample
raw = np.load(NPZ_PATH)['data']
idx = np.random.default_rng(42).choice(len(raw), SUBSAMPLE, replace=False)
data = raw[idx]
print(f"Using {len(data):,} rows (subsampled from {len(raw):,})")

# Move entire dataset to GPU once — avoids per-batch transfer bottleneck
X = torch.tensor(data, dtype=torch.float32).to(DEVICE)
loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=True)

class VanillaAE(nn.Module):
    def __init__(self, in_dim=18):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),    nn.ReLU(),
            nn.Linear(32, 16),    nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),    nn.ReLU(),
            nn.Linear(32, 64),    nn.ReLU(),
            nn.Linear(64, in_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model     = VanillaAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
scaler    = torch.cuda.amp.GradScaler()   # AMP

print("Training Vanilla AE...")
start = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for (batch,) in loader:
        with torch.cuda.amp.autocast():
            loss = criterion(model(batch), batch)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item() * len(batch)
    if epoch % 5 == 0:
        print(f"  Epoch {epoch}/{EPOCHS}  loss={epoch_loss/len(X):.6f}")

train_time = time.perf_counter() - start

model.eval()
all_scores = []
with torch.no_grad():
    for (batch,) in DataLoader(TensorDataset(X), batch_size=8192):
        with torch.cuda.amp.autocast():
            recon = model(batch)
        mse = ((batch - recon) ** 2).mean(dim=1)
        all_scores.append(mse.cpu().numpy())

scores = np.concatenate(all_scores)
threshold = np.percentile(scores, THRESHOLD_PERCENTILE)
anomaly_rate = (scores > threshold).sum() / len(scores) * 100

print(f"\n{'='*45}")
print(f"Model          : Vanilla Autoencoder (Dense AE)")
print(f"Train time     : {train_time:.1f}s")
print(f"Mean MSE       : {scores.mean():.6f}")
print(f"Threshold (p95): {threshold:.6f}")
print(f"Anomaly rate   : {anomaly_rate:.2f}%")
print(f"{'='*45}")

np.save(os.path.join(OUT_DIR, "vanilla_ae_scores.npy"), scores)
print(f"Scores saved → {OUT_DIR}/vanilla_ae_scores.npy")
