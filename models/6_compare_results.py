"""
Run AFTER all 5 model scripts.
Produces a full technical comparison table suitable for the paper:
  - Model parameters count
  - Training time
  - Inference time per sample (ms)
  - Mean/Std of anomaly scores
  - Anomaly threshold (p95)
  - Anomaly rate %
  - Score separability ratio (anomaly mean / normal mean)
  - Temporal awareness, architecture type, framework
Also runs Isolation Forest + One-Class SVM baselines.
"""

import numpy as np
import time, os
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

NPZ_PATH  = r"c:\Users\bharg\OneDrive\Documents\GPU_package\data\cesnet_10min.npz"
OUT_DIR   = r"c:\Users\bharg\OneDrive\Documents\GPU_package\results"
SUBSAMPLE = 3_000_000
THRESHOLD_PERCENTILE = 95

os.makedirs(OUT_DIR, exist_ok=True)

raw_full = np.load(NPZ_PATH)['data']
idx = np.random.default_rng(42).choice(len(raw_full), SUBSAMPLE, replace=False)
raw = raw_full[idx]
print(f"Loaded {len(raw):,} rows for classical baselines\n")

# ── helpers ───────────────────────────────────────────────────────────────────
def score_stats(scores, label=""):
    th       = np.percentile(scores, THRESHOLD_PERCENTILE)
    mask_anom = scores > th
    mask_norm = ~mask_anom
    anom_rate = mask_anom.sum() / len(scores) * 100
    sep_ratio = scores[mask_anom].mean() / scores[mask_norm].mean() if mask_norm.sum() > 0 else float('nan')
    return {
        "mean"      : scores.mean(),
        "std"       : scores.std(),
        "min"       : scores.min(),
        "p50"       : np.percentile(scores, 50),
        "p95"       : th,
        "p99"       : np.percentile(scores, 99),
        "max"       : scores.max(),
        "anom_rate" : anom_rate,
        "sep_ratio" : sep_ratio,
    }

results = []   # (name, arch_type, temporal, supervised, framework, n_params, train_s, infer_us, stats)

# ── 1. Isolation Forest ───────────────────────────────────────────────────────
print("Running Isolation Forest...")
t0 = time.perf_counter()
IF = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
IF.fit(raw)
train_t = time.perf_counter() - t0

t0 = time.perf_counter()
if_scores = -IF.score_samples(raw)
infer_t = (time.perf_counter() - t0) / len(raw) * 1e6   # µs per sample

results.append(("Isolation Forest", "Tree Ensemble", "No", "No", "sklearn",
                "N/A", train_t, infer_t, score_stats(if_scores)))
print(f"  Done. Anomaly rate: {score_stats(if_scores)['anom_rate']:.2f}%")

# ── 2. One-Class SVM ─────────────────────────────────────────────────────────
print("Running One-Class SVM (subsampled to 20k for fit)...")
t0 = time.perf_counter()
fit_idx = np.random.default_rng(42).choice(len(raw), 20000, replace=False)
svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
svm.fit(raw[fit_idx])
train_t = time.perf_counter() - t0

t0 = time.perf_counter()
svm_scores = -svm.score_samples(raw)
infer_t = (time.perf_counter() - t0) / len(raw) * 1e6

results.append(("One-Class SVM", "Kernel Boundary", "No", "No", "sklearn",
                "N/A", train_t, infer_t, score_stats(svm_scores)))
print(f"  Done. Anomaly rate: {score_stats(svm_scores)['anom_rate']:.2f}%")

# ── 3–7. Load PyTorch model scores ────────────────────────────────────────────
pytorch_models = [
    ("Vanilla AE (Dense)",        "vanilla_ae_scores.npy",        "Dense AE",        "No",  "PyTorch"),
    ("Simple LSTM-AE",            "lstm_ae_simple_scores.npy",    "LSTM AE",         "Yes", "PyTorch"),
    ("TCN Autoencoder",           "tcn_ae_scores.npy",            "Conv1D AE",       "Yes", "PyTorch"),
    ("Transformer AE",            "transformer_ae_scores.npy",    "Transformer AE",  "Yes", "PyTorch"),
    ("Bidir LSTM-AE+Attn (Ours)", "bidir_lstm_ae_scores.npy",     "Bidir LSTM AE",   "Yes", "PyTorch"),
]

# Approximate param counts (computed analytically)
param_counts = {
    "vanilla_ae_scores.npy"      : "~6K",
    "lstm_ae_simple_scores.npy"  : "~330K",
    "tcn_ae_scores.npy"          : "~105K",
    "transformer_ae_scores.npy"  : "~95K",
    "bidir_lstm_ae_scores.npy"   : "~660K",
}

for name, fname, arch, temporal, fw in pytorch_models:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        sc = np.load(fpath)
        st = score_stats(sc)
        results.append((name, arch, temporal, "No", fw,
                        param_counts[fname], "-", "-", st))
        print(f"  {name}: anomaly rate {st['anom_rate']:.2f}%  sep_ratio {st['sep_ratio']:.2f}x")
    else:
        print(f"  {name}: NOT RUN — missing {fpath}")
        results.append((name, arch, temporal, "No", fw,
                        param_counts[fname], "NOT RUN", "NOT RUN", None))

# ── Print full technical table ────────────────────────────────────────────────
SEP = "=" * 130
print(f"\n\n{SEP}")
print("COMPREHENSIVE MODEL COMPARISON — CESNET Time-Series 24 (10-min aggregation)")
print(f"{SEP}")

# Table 1: Architecture & Complexity
print("\nTABLE A — Architecture & Computational Complexity")
print("-" * 90)
print(f"{'Model':<30} {'Architecture':<18} {'Temporal':<10} {'Params':<10} {'Train(s)':<12} {'Infer(µs/sample)':<18}")
print("-" * 90)
for r in results:
    name, arch, temp, sup, fw, params, train_t, infer_t, st = r
    tr = f"{train_t:.1f}" if isinstance(train_t, float) else train_t
    it = f"{infer_t:.2f}" if isinstance(infer_t, float) else infer_t
    print(f"{name:<30} {arch:<18} {temp:<10} {params:<10} {tr:<12} {it:<18}")

# Table 2: Detection Performance
print(f"\n\nTABLE B — Anomaly Detection Performance")
print("-" * 110)
print(f"{'Model':<30} {'Mean Score':<12} {'Std Score':<11} {'p50':<10} {'p95 (thresh)':<14} {'p99':<10} {'Anom%':<9} {'Sep.Ratio':<10}")
print("-" * 110)
for r in results:
    name, arch, temp, sup, fw, params, train_t, infer_t, st = r
    if st is None:
        print(f"{name:<30} {'NOT RUN'}")
        continue
    print(f"{name:<30} {st['mean']:<12.6f} {st['std']:<11.6f} {st['p50']:<10.6f} "
          f"{st['p95']:<14.6f} {st['p99']:<10.6f} {st['anom_rate']:<9.2f} {st['sep_ratio']:<10.2f}")

print(f"\n{SEP}")
print("NOTES:")
print("  - All models evaluated on same 3M-row subsample of CESNET agg_10_minutes")
print("  - Preprocessing: log1p transform + MinMaxScaler (identical for all models)")
print("  - Anomaly threshold = 95th percentile of reconstruction/anomaly scores")
print("  - Sep.Ratio = mean(anomaly scores) / mean(normal scores)  [higher = better separation]")
print("  - One-Class SVM fitted on 20k subsample due to O(n²) complexity")
print("  - Params: approximate learnable parameter count")
print(f"{SEP}\n")
