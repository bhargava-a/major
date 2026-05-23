CESNET Anomaly Detection - Model Comparison Package (v2, OPTIMIZED)
====================================================================

What changed in v2
------------------
  1. New model: bilstm_attn_opt
       Optimized BiLSTM+Attention autoencoder. Targets the reconstruction-
       quality and speed gap the first run exposed (old bilstm_attn was
       slowest and among worst on test MSE).

       Architecture changes vs. original bilstm_attn:
         - Per-timestep representation kept end-to-end (no single-vector
           pooling bottleneck - this was the main cause of poor results).
         - Input projection to d_model + learnable positional embedding.
         - 1-layer BiLSTM (2 layers removed, no benefit, just cost).
         - Multi-head self-attention block refines per-timestep encoding.
         - Pre-norm residual blocks + dropout 0.1.
         - Residual skip from input to output so the model only learns
           the delta from identity - lower effective loss, faster training.

  2. Training upgrades applied to ALL models:
         - AdamW (Adam + weight decay 1e-5) instead of plain Adam.
         - OneCycleLR schedule with warmup + cosine anneal (max_lr = 3*lr).
         - Mixed precision (AMP) on CUDA - faster steps, less memory.
         - Gradient clipping at 1.0 (kept).

  3. All previous models kept for apples-to-apples comparison.

Contents
--------
  data/cesnet_subset.npz     50 entities, 8 weeks of 10-min CESNET traffic
                             403,200 rows, 18 features
                             splits: 302,400 train / 50,400 val / 50,400 test
  train_all_models.py        single-file runner for all 6 models + plots
  requirements.txt           python deps: torch, numpy, matplotlib
  README.txt                 this file

Models
------
  1. vanilla_ae         MLP autoencoder on flattened window (baseline)
  2. lstm_ae            Unidirectional 2-layer LSTM encoder-decoder
  3. tcn_ae             Temporal Convolutional Network autoencoder
  4. transformer_ae     Self-attention transformer autoencoder
  5. bilstm_attn        Original BiLSTM+Attention (reference, your old model)
  6. bilstm_attn_opt    OPTIMIZED BiLSTM+Attention (new primary model)

Setup
-----
  pip install -r requirements.txt

GPU (recommended):
  pip install torch --index-url https://download.pytorch.org/whl/cu121

Run
---
Train all 6 models with default paper settings:
  python train_all_models.py

Just the new optimized model (fastest iteration):
  python train_all_models.py --only bilstm_attn_opt

Compare only old vs new BiLSTM+Attn:
  python train_all_models.py --only bilstm_attn
  python train_all_models.py --only bilstm_attn_opt

CLI flags
---------
  --epochs 50          max training epochs
  --patience 4         early stop on val loss plateau
  --batch-size 256     raise to 512/1024 on GPUs with spare memory
  --lr 1e-3            base LR (OneCycleLR peaks at 3x this)
  --window 10          sliding window length
  --seed 42
  --no-amp             disable mixed precision (debugging only)
  --no-sched           disable OneCycleLR schedule

Outputs (./results/)
--------------------
  results.txt                         full terminal log
  results.csv                         one row per model with all metrics
  results.json                        per-epoch loss histories
  plot_loss_curves.png                per-model train/val curves (log scale)
  plot_val_overlay.png                all models' val loss on one axis
  plot_summary_bars.png               bars: params, train time, val/test MSE,
                                        inference ms/batch, us/sample
  plot_recon_error_distribution.png   test-set per-window MSE histograms
  plot_anomaly_rate.png               anomaly flag rate vs threshold
  plot_params_vs_loss.png             capacity vs reconstruction quality

Note on the data
----------------
  Sliding windows are built WITHIN each entity only (no boundary crossing).
  Temporal split per entity: first 6 weeks train, week 7 val, week 8 test.
  Features are already log1p + MinMax scaled. Do NOT re-scale.

Note on MSE vs. real anomaly detection
--------------------------------------
  Reconstruction MSE rewards a model that perfectly reproduces ALL input,
  including anomalies. A model with slightly worse MSE can be a better
  anomaly detector if the gap between normal and anomalous reconstruction
  errors is bigger. Check plot_recon_error_distribution.png and
  plot_anomaly_rate.png - those are the anomaly-detection-relevant views.
