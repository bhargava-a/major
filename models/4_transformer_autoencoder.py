"""
Model 4: Transformer Autoencoder
Reference: "Unsupervised Anomaly Detection and Explanation in Network Traffic with Transformers" (MDPI Electronics 2024)
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
BATCH_SIZE = 512
EPOCHS     = 20
LR         = 1e-3
D_MODEL    = 64
N_HEADS    = 2
N_LAYERS   = 2
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

class TransformerAE(nn.Module):
    def __init__(self, n_features=18, d_model=64, n_heads=2, n_layers=2, window=10):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128,
                                               dropout=0.1, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=128,
                                               dropout=0.1, batch_first=True)
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, n_features)
        self.pos_emb  = nn.Embedding(window, d_model)
        self.register_buffer('pos_idx', torch.arange(window))

    def forward(self, x):
        pos    = self.pos_emb(self.pos_idx).unsqueeze(0)
        x_proj = self.input_proj(x) + pos
        memory = self.encoder(x_proj)
        out    = self.decoder(x_proj, memory)
        return self.out_proj(out)

model     = TransformerAE(window=WINDOW).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
scaler    = torch.cuda.amp.GradScaler()

print("Training Transformer Autoencoder...")
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
    for batch in DataLoader(dataset, batch_size=1024, num_workers=0, pin_memory=True):
        batch = batch.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            recon = model(batch)
        mse = ((batch - recon) ** 2).mean(dim=(1, 2))
        all_scores.append(mse.cpu().numpy())

scores = np.concatenate(all_scores)
threshold = np.percentile(scores, THRESHOLD_PERCENTILE)
anomaly_rate = (scores > threshold).sum() / len(scores) * 100

print(f"\n{'='*45}")
print(f"Model          : Transformer Autoencoder")
print(f"Train time     : {train_time:.1f}s")
print(f"Mean MSE       : {scores.mean():.6f}")
print(f"Threshold (p95): {threshold:.6f}")
print(f"Anomaly rate   : {anomaly_rate:.2f}%")
print(f"{'='*45}")

np.save(os.path.join(OUT_DIR, "transformer_ae_scores.npy"), scores)
print(f"Scores saved → {OUT_DIR}/transformer_ae_scores.npy")
