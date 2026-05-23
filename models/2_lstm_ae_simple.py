"""
Model 2: Simple LSTM Autoencoder
Reference: Malhotra et al. (2015) — "Long Short Term Memory Networks for Anomaly Detection in Time Series"
"""
import numpy as np
import torch
import torch.nn as nn
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
idx.sort()   # keep temporal order for windowing
data    = raw[idx]
dataset = WindowDataset(data, WINDOW)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=0, pin_memory=True)
print(f"Using {len(data):,} rows → {len(dataset):,} windows")

class SimpleLSTMAE(nn.Module):
    def __init__(self, n_features=18, hidden=128, n_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, n_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, n_layers, batch_first=True)
        self.fc_out  = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        dec_in = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(dec_in)
        return self.fc_out(out)

model     = SimpleLSTMAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
scaler    = torch.cuda.amp.GradScaler()

print("Training Simple LSTM-AE...")
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
print(f"Model          : Simple LSTM Autoencoder")
print(f"Train time     : {train_time:.1f}s")
print(f"Mean MSE       : {scores.mean():.6f}")
print(f"Threshold (p95): {threshold:.6f}")
print(f"Anomaly rate   : {anomaly_rate:.2f}%")
print(f"{'='*45}")

np.save(os.path.join(OUT_DIR, "lstm_ae_simple_scores.npy"), scores)
print(f"Scores saved → {OUT_DIR}/lstm_ae_simple_scores.npy")
