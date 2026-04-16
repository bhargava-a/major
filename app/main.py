"""
FastAPI application — entry point for the observability system.

Endpoints:
  GET  /          → HTML status page
  GET  /metrics   → Prometheus metrics (scraped by Prometheus every 5s)
  GET  /api/anomalies       → Last 50 anomalies from SQLite
  GET  /api/anomalies/count → Total anomaly count
  GET  /api/status          → Live system status (score, is_anomaly, etc.)
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

from app.model_inference import AnomalyDetector
from app.database import init_db, get_recent_anomalies
from app.ingestor import run_ingestor
import app.metrics as m
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

detector = AnomalyDetector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    task = asyncio.create_task(run_ingestor(detector))
    yield
    task.cancel()


app = FastAPI(title="AI Observability System", lifespan=lifespan)


# ── Prometheus metrics endpoint ─────────────────────────────────────────────
@app.get("/metrics")
def metrics():
    """Prometheus scrapes this every 5 seconds."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── REST API ─────────────────────────────────────────────────────────────────
@app.get("/api/anomalies")
def list_anomalies(limit: int = 50):
    return get_recent_anomalies(limit)


@app.get("/api/anomalies/count")
def anomaly_count():
    rows = get_recent_anomalies(limit=10_000)
    return {"total": len(rows)}


@app.get("/api/status")
def status():
    return {
        "anomaly_score": m.ANOMALY_SCORE._value.get(),
        "is_anomaly": bool(m.IS_ANOMALY._value.get()),
        "samples_processed": int(m.SAMPLES_PROCESSED._value.get()),
        "anomalies_total": int(m.ANOMALIES_TOTAL._value.get()),
    }


# ── Simple HTML dashboard (no Grafana needed for a quick look) ───────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>AI Observability — Network Anomaly Detection</title>
  <meta http-equiv="refresh" content="3">
  <style>
    body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 30px; }
    h1   { color: #58a6ff; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
            padding: 20px; margin: 15px 0; }
    .anomaly { color: #f85149; font-weight: bold; }
    .normal  { color: #3fb950; font-weight: bold; }
    table  { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px 12px; border: 1px solid #30363d; text-align: left; }
    th     { background: #21262d; }
    .badge-high   { color: #f85149; }
    .badge-medium { color: #e3b341; }
    .badge-low    { color: #3fb950; }
  </style>
  <script>
    async function loadData() {
      const s = await fetch('/api/status').then(r => r.json());
      document.getElementById('score').textContent = s.anomaly_score.toFixed(6);
      document.getElementById('flag').textContent = s.is_anomaly ? 'ANOMALY' : 'NORMAL';
      document.getElementById('flag').className = s.is_anomaly ? 'anomaly' : 'normal';
      document.getElementById('total').textContent = s.anomalies_total;
      document.getElementById('processed').textContent = s.samples_processed;

      const rows = await fetch('/api/anomalies?limit=20').then(r => r.json());
      const tbody = document.getElementById('tbody');
      tbody.innerHTML = rows.map(r => `
        <tr>
          <td>${r.detected_at.replace('T',' ').slice(0,19)}</td>
          <td>${r.anomaly_score.toFixed(6)}</td>
          <td class="badge-${r.severity.toLowerCase()}">${r.severity}</td>
          <td>${(r.n_flows/1e6).toFixed(2)}M</td>
          <td>${(r.n_bytes/1e9).toFixed(2)}GB</td>
        </tr>`).join('');
    }
    loadData();
    setInterval(loadData, 3000);
  </script>
</head>
<body>
  <h1>AI-Augmented Observability System</h1>
  <div class="card">
    <h3>Live Status (auto-refresh 3s)</h3>
    <p>Current Score: <strong id="score">—</strong> &nbsp;|&nbsp;
       Status: <strong id="flag">—</strong></p>
    <p>Samples Processed: <strong id="processed">—</strong> &nbsp;|&nbsp;
       Anomalies Detected: <strong id="total">—</strong></p>
  </div>
  <div class="card">
    <h3>Recent Anomalies</h3>
    <table>
      <thead><tr><th>Time</th><th>Score</th><th>Severity</th><th>n_flows</th><th>n_bytes</th></tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
  <div class="card">
    <p>Grafana dashboard: <a href="http://localhost:3000" style="color:#58a6ff">http://localhost:3000</a></p>
    <p>Prometheus: <a href="http://localhost:9090" style="color:#58a6ff">http://localhost:9090</a></p>
    <p>Raw metrics: <a href="/metrics" style="color:#58a6ff">/metrics</a></p>
  </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML
