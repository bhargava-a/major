"""Prometheus metric definitions — imported by ingestor and main."""
from prometheus_client import Gauge, Counter

# ── Model output ─────────────────────────────────────────────────────────────
ANOMALY_SCORE = Gauge("network_anomaly_score", "LSTM reconstruction error (anomaly score)")
ANOMALY_THRESHOLD = Gauge("network_anomaly_threshold", "Trained anomaly detection threshold")
IS_ANOMALY = Gauge("network_is_anomaly", "1 = anomaly detected, 0 = normal")
ANOMALIES_TOTAL = Counter("network_anomalies_total", "Total anomalies detected since startup")
SAMPLES_PROCESSED = Counter("network_samples_processed_total", "Total samples processed by model")

# ── Traffic volume ───────────────────────────────────────────────────────────
N_FLOWS   = Gauge("network_n_flows",   "Number of network flows in current window")
N_PACKETS = Gauge("network_n_packets", "Number of packets in current window")
N_BYTES   = Gauge("network_n_bytes",   "Number of bytes in current window")

# ── Protocol behaviour ───────────────────────────────────────────────────────
TCP_UDP_RATIO_PACKETS = Gauge("network_tcp_udp_ratio_packets", "TCP to UDP packet ratio (>1 = TCP dominant)")
TCP_UDP_RATIO_BYTES   = Gauge("network_tcp_udp_ratio_bytes",   "TCP to UDP byte ratio")
DIR_RATIO_PACKETS     = Gauge("network_dir_ratio_packets",     "Inbound vs outbound packet ratio (asymmetry = suspicious)")
DIR_RATIO_BYTES       = Gauge("network_dir_ratio_bytes",       "Inbound vs outbound byte ratio")

# ── Destination diversity ─────────────────────────────────────────────────────
AVG_DEST_IP    = Gauge("network_avg_dest_ip",    "Avg unique destination IPs per flow (spikes on scans)")
AVG_DEST_PORTS = Gauge("network_avg_dest_ports", "Avg unique destination ports per flow (spikes on port scans)")
AVG_DEST_ASN   = Gauge("network_avg_dest_asn",   "Avg unique destination ASNs per flow")

# ── Flow characteristics ─────────────────────────────────────────────────────
AVG_DURATION = Gauge("network_avg_duration", "Average flow duration in seconds")
AVG_TTL      = Gauge("network_avg_ttl",      "Average IP Time-To-Live value")
