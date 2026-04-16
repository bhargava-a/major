"""
Load the saved LSTM Autoencoder and run inference on a sliding window.
The model was saved with keys: model_state_dict, threshold, scaler_min, scaler_max
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import deque

MODEL_PATH = Path(__file__).parent.parent / "lstm_autoencoder_model.pth"

# Same feature columns used during training (18 features, columns 1-18)
FEATURE_NAMES = [
    "n_flows", "n_packets", "n_bytes",
    "sum_n_dest_asn", "average_n_dest_asn", "std_n_dest_asn",
    "sum_n_dest_ports", "average_n_dest_ports", "std_n_dest_ports",
    "sum_n_dest_ip", "average_n_dest_ip", "std_n_dest_ip",
    "tcp_udp_ratio_packets", "tcp_udp_ratio_bytes",
    "dir_ratio_packets", "dir_ratio_bytes",
    "avg_duration", "avg_ttl",
]
SKEWED_INDICES = [0, 1, 2]  # n_flows, n_packets, n_bytes → log1p
WINDOW_SIZE = 10
INPUT_DIM = len(FEATURE_NAMES)  # 18


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, encoder_outputs):
        weights = F.softmax(self.attention(encoder_outputs), dim=1)
        context = torch.sum(weights * encoder_outputs, dim=1)
        return context, weights


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0, bidirectional=True,
        )
        self.attention = Attention(hidden_dim * 2)
        self.fc_compress = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, input_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        encoder_outputs, _ = self.encoder(x)
        context, _ = self.attention(encoder_outputs)
        compressed = self.layer_norm(F.relu(self.fc_compress(context)))
        decoder_input = compressed.unsqueeze(1).repeat(1, seq_len, 1)
        h_dec = compressed.unsqueeze(0).repeat(self.n_layers, 1, 1)
        c_dec = torch.zeros_like(h_dec)
        decoded, _ = self.decoder(decoder_input, (h_dec, c_dec))
        return self.output_layer(decoded)


class AnomalyDetector:
    """Wraps the trained model with preprocessing and a sliding window buffer."""

    def __init__(self):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        self.threshold: float = float(checkpoint["threshold"])
        self.scaler_min: np.ndarray = checkpoint["scaler_min"].astype(np.float32)
        self.scaler_scale: np.ndarray = (
            checkpoint["scaler_max"] - checkpoint["scaler_min"]
        ).astype(np.float32)
        # Avoid divide-by-zero for constant features
        self.scaler_scale[self.scaler_scale == 0] = 1.0

        self.model = LSTMAutoencoder(input_dim=INPUT_DIM)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Sliding window buffer: holds last WINDOW_SIZE preprocessed feature vectors
        self._buffer: deque = deque(maxlen=WINDOW_SIZE)

    def _preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Apply log1p + MinMaxScaler to a single feature vector (18,)."""
        x = raw.copy().astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        for i in SKEWED_INDICES:
            x[i] = np.log1p(x[i])
        x = (x - self.scaler_min) / self.scaler_scale
        return x

    def push(self, row: dict) -> dict | None:
        """
        Push one raw data row (dict of feature_name → value).
        Returns a result dict when the window is full, else None.
        """
        raw = np.array([row[f] for f in FEATURE_NAMES], dtype=np.float32)
        processed = self._preprocess(raw)
        self._buffer.append(processed)

        if len(self._buffer) < WINDOW_SIZE:
            return None  # Not enough data yet

        window = np.array(list(self._buffer), dtype=np.float32)  # (10, 18)
        tensor = torch.FloatTensor(window).unsqueeze(0)  # (1, 10, 18)

        with torch.no_grad():
            reconstruction = self.model(tensor)
            mse = F.mse_loss(reconstruction, tensor).item()

        is_anomaly = mse > self.threshold
        return {
            "anomaly_score": round(mse, 6),
            "is_anomaly": bool(is_anomaly),
            "threshold": round(self.threshold, 6),
            # Traffic volume
            "n_flows":   float(row["n_flows"]),
            "n_packets": float(row["n_packets"]),
            "n_bytes":   float(row["n_bytes"]),
            # Protocol behaviour
            "tcp_udp_ratio_packets": float(row["tcp_udp_ratio_packets"]),
            "tcp_udp_ratio_bytes":   float(row["tcp_udp_ratio_bytes"]),
            "dir_ratio_packets":     float(row["dir_ratio_packets"]),
            "dir_ratio_bytes":       float(row["dir_ratio_bytes"]),
            # Destination diversity
            "average_n_dest_ip":    float(row["average_n_dest_ip"]),
            "average_n_dest_ports": float(row["average_n_dest_ports"]),
            "average_n_dest_asn":   float(row["average_n_dest_asn"]),
            # Flow characteristics
            "avg_duration": float(row["avg_duration"]),
            "avg_ttl":      float(row["avg_ttl"]),
        }
