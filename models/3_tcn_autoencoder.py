"""
Model 3: TCN Autoencoder (Temporal Convolutional Network)
Reference: "A Novel Unsupervised Anomaly Detection Method Based on TCN-LSTM-CMA Autoencoder" (Springer 2024)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time, os

NPZ_PATH   = r"c:\Users\bharg\OneDrive\Documents\GPU_package\data\cesnet_10min.npz"
OUT_DIR    = r"c:\Users\bharg\OneDrive\Documents\GPU_package\results"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW     = 10
BATCH_SIZE = 1024
EPOCHS     = 20
LR         = 1e-3
SUBSAMPLE  = 3_000_000
THRESHOLD_PERCENTILE = 95

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

class WindowDataset(Dataset):
    def __init__(self, data, window):
        self.data   = torch.tensor(data, dtype=torch.float32)
        self.window = window
    def __len__(self):
        return len(self.data) - self.window + 1
    def __getitem__(self, i):
        return self.data[i:i + self.window]

raw = np.load(NPZ_PATH)['data']
idx = np.random.default_rng(42).choice(len(raw), SUBSAMPLE, replace=False)
idx.sort()
data    = raw[idx]
dataset = WindowDataset(data, WINDOW)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=0, pin_memory=True)
print(f"Using {len(data):,} rows → {len(dataset):,} windows")

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1):
        super().__init__()
        self.pad   = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=self.pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=self.pad, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.res   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        if self.pad: out = out[..., :-self.pad]
        out = F.relu(self.norm1(out))
        out = self.conv2(out)
        if self.pad: out = out[..., :-self.pad]
        out = F.relu(self.norm2(out))
        return out + self.res(x)

class TCNAE(nn.Module):
    def __init__(self, n_features=18):
        super().__init__()
        self.encoder = nn.Sequential(
            TCNBlock(n_features, 64, dilation=1),
            TCNBlock(64,         32, dilation=2),
            TCNBlock(32,         16, dilation=4),
        )
        self.decoder = nn.Sequential(
            TCNBlock(16,         32, dilation=4),
            TCNBlock(32,         64, dilation=2),
            TCNBlock(64, n_features, dilation=1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)       # (B, F, T)
        z = self.encoder(x)
        return self.decoder(z).transpose(1, 2)   # (B, T, F)

model     = TCNAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
scaler    = torch.cuda.amp.GradScaler()

print("Training TCN Autoencoder...")
start = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss = criterion(model(batch), batch)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item() * len(batch)
    if epoch % 5 == 0:
        print(f"  Epoch {epoch}/{EPOCHS}  loss={epoch_loss/len(dataset):.6f}")

train_time = time.perf_counter() - start

model.eval()
all_scores = []
with torch.no_grad():
    for batch in DataLoader(dataset, batch_size=2048, num_workers=0, pin_memory=True):
        batch = batch.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            recon = model(batch)
        mse = ((batch - recon) ** 2).mean(dim=(1, 2))
        all_scores.append(mse.cpu().numpy())

scores = np.concatenate(all_scores)
threshold = np.percentile(scores, THRESHOLD_PERCENTILE)
anomaly_rate = (scores > threshold).sum() / len(scores) * 100

print(f"\n{'='*45}")
print(f"Model          : TCN Autoencoder")
print(f"Train time     : {train_time:.1f}s")
print(f"Mean MSE       : {scores.mean():.6f}")
print(f"Threshold (p95): {threshold:.6f}")
print(f"Anomaly rate   : {anomaly_rate:.2f}%")
print(f"{'='*45}")

np.save(os.path.join(OUT_DIR, "tcn_ae_scores.npy"), scores)
print(f"Scores saved → {OUT_DIR}/tcn_ae_scores.npy")
