# 🚀 AI Observability System - Quick Start Guide

**Fast-track setup for running the main FastAPI application only** (no notebooks)

---

## 📋 Prerequisites

Make sure you have installed:

- **Python 3.8+** → [Download](https://www.python.org/downloads/)
- **Git** → [Download](https://git-scm.com/)
- **Docker & Docker Compose** (optional, for Prometheus/Grafana) → [Download](https://www.docker.com/products/docker-desktop)

Verify installations:
```powershell
python --version
git --version
```

---

## ⚡ Quick Setup (5 minutes)

### Step 1: Clone Repository
```powershell
git clone https://github.com/bhargava-a/major.git
cd major
```

### Step 2: Create Virtual Environment
```powershell
python -m venv venv

# Activate (Windows PowerShell):
.\venv\Scripts\Activate.ps1

# On CMD, use:
# venv\Scripts\activate.bat
```

### Step 3: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `torch` - PyTorch (ML models)
- `prometheus-client` - Metrics
- `pandas`, `numpy` - Data processing
- `sqlalchemy` - Database ORM

### Step 4: Run the Application
```powershell
# With venv activated:
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

✅ **API is now running at: `http://localhost:8000`**

---

## 📡 Verify It Works

Open a new PowerShell terminal and test:

```powershell
# Check if API is alive:
$response = Invoke-WebRequest http://localhost:8000/api/status
$response.Content | ConvertFrom-Json | Format-Table

# Get recent anomalies:
$response = Invoke-WebRequest http://localhost:8000/api/anomalies
$response.Content | ConvertFrom-Json | Format-Table

# View Prometheus metrics:
Invoke-WebRequest http://localhost:8000/metrics
```

---

## 📚 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| **GET** | `/` | HTML status page |
| **GET** | `/metrics` | Prometheus metrics (scraped every 5s) |
| **GET** | `/api/status` | Current anomaly score & status |
| **GET** | `/api/anomalies` | Last 50 anomalies from database |
| **GET** | `/api/anomalies?limit=100` | Get 100 most recent anomalies |
| **GET** | `/api/anomalies/count` | Total anomaly count |

### Example API Calls

```powershell
# Get live status
curl http://localhost:8000/api/status

# Get last 50 anomalies
curl http://localhost:8000/api/anomalies

# Get specific limit
curl "http://localhost:8000/api/anomalies?limit=20"

# View Prometheus metrics
curl http://localhost:8000/metrics
```

### Interactive API Docs
Open in browser: **`http://localhost:8000/docs`**
- Swagger UI with live testing
- Schema documentation

---

## 🐳 Optional: Run with Docker

### Prerequisites
- Docker Desktop installed and running

### Start Prometheus & Grafana Only

```powershell
# Start services:
docker-compose up -d

# View logs:
docker-compose logs -f

# Stop services:
docker-compose down
```

This starts:
- **Prometheus** (metrics storage): `http://localhost:9090`
- **Grafana** (dashboards): `http://localhost:3000`

### Run FastAPI Backend (in separate terminal)

```powershell
# With venv activated:
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

---

## 🎯 Architecture Overview

```
┌─────────────────┐
│   FastAPI App   │
│  (port 8000)    │
│                 │
├─ Ingestor      │ ← Collects traffic data
├─ ML Detector   │ ← LSTM Autoencoder model
├─ Database      │ ← SQLite (anomalies)
└─ Metrics       │ ← Prometheus metrics
        │
        ├──→ Prometheus (port 9090) │ Optional
        │                           │ if Docker
        ├──→ Grafana (port 3000)    │ is used
```

---

## 🔧 Configuration

### Environment Variables (Optional)

Create a `.env` file in project root:

```env
# Application
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///./anomalies.db

# Model
MODEL_PATH=lstm_autoencoder_model.pth
DEVICE=cpu  # or 'cuda' for GPU

# Metrics
PROMETHEUS_PORT=8000
METRICS_ENABLED=true
```

Load in application:
```powershell
# Install python-dotenv:
pip install python-dotenv
```

### Model Path

Default: `lstm_autoencoder_model.pth` (must be in project root)

If missing, check [app/model_inference.py](app/model_inference.py) for details on retraining.

---

## 📊 Data Files

| File | Purpose |
|------|---------|
| `lstm_autoencoder_model.pth` | Pre-trained ML model (required) |
| `anomalies.db` | SQLite database (auto-created) |
| `sample_traffic_data.csv` | Sample dataset for testing |

---

## 🚨 Troubleshooting

### Port 8000 Already in Use
```powershell
# Find process using port 8000:
Get-NetTCPConnection -LocalPort 8000

# Kill it (replace PID):
Stop-Process -Id <PID> -Force

# Or use different port:
python -m uvicorn app.main:app --port 8001
```

### PyTorch Installation Issues
```powershell
# For CPU-only (smaller):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU (if CUDA 11.8 installed):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Module Not Found Errors
```powershell
# Ensure venv is activated (should show (venv) in prompt)
# Reinstall dependencies:
pip install -r requirements.txt --force-reinstall
```

### Database Lock Error
```powershell
# Delete and recreate:
Remove-Item anomalies.db
# Restart the application
```

---

## 📁 Project Structure

```
major/
├── app/
│   ├── main.py                 ← FastAPI entry point
│   ├── model_inference.py      ← PyTorch model inference
│   ├── ingestor.py             ← Data ingestion logic
│   ├── database.py             ← SQLite operations
│   ├── metrics.py              ← Prometheus metrics
│   └── simulator.py            ← Traffic simulation
├── lstm_autoencoder_model.pth  ← Pre-trained model
├── requirements.txt            ← Dependencies
├── docker-compose.yml          ← Docker services
├── prometheus.yml              ← Prometheus config
├── sample_traffic_data.csv     ← Sample data
└── README_QUICKSTART.md        ← This file
```

---

## 🎯 Common Tasks

### View Live Metrics
```powershell
# Continuously poll metrics:
while ($true) {
    (Invoke-WebRequest http://localhost:8000/api/status).Content | ConvertFrom-Json
    Start-Sleep -Seconds 2
}
```

### Export Anomalies to CSV
```powershell
$anomalies = (Invoke-WebRequest http://localhost:8000/api/anomalies).Content | ConvertFrom-Json
$anomalies | Export-Csv -Path anomalies_export.csv -NoTypeInformation
```

### Check Database Size
```powershell
(Get-Item anomalies.db).Length / 1MB  # in MB
```

### Clear Anomalies Database
```powershell
Remove-Item anomalies.db
# Restart application
```

---

## 🔒 Security Notes

- Default Grafana: `admin/admin` (change in production)
- SQLite database stored locally (use PostgreSQL for production)
- API has no authentication (add in production)
- Model runs on CPU by default (use GPU for faster inference)

---

## 📈 Performance Tips

1. **Enable GPU**: 
   - Ensure CUDA toolkit installed
   - Update PyTorch for GPU
   - Set `DEVICE=cuda` in config

2. **Increase Batch Size**:
   - Edit [app/ingestor.py](app/ingestor.py)
   - Adjust `batch_size` parameter

3. **Database Optimization**:
   - Regular database cleanup
   - Consider PostgreSQL for production

4. **Metrics Retention**:
   - Edit `prometheus.yml` for retention policy
   - Default: 7 days

---

## ✅ Success Checklist

- [ ] Git cloned
- [ ] Python venv created and activated
- [ ] Dependencies installed
- [ ] FastAPI running without errors
- [ ] API accessible at `http://localhost:8000/docs`
- [ ] `/api/status` returns data
- [ ] `/api/anomalies` returns list
- [ ] Prometheus metrics scraped (check `/metrics`)

---

## 🆘 Getting Help

1. Check logs: Look for errors in terminal output
2. Test endpoint: `http://localhost:8000/api/status`
3. View API docs: `http://localhost:8000/docs`
4. Check database: `anomalies.db` should exist after first run

---

## 📞 Support

For issues or questions:
1. Review this guide
2. Check [README.md](README.md) for detailed documentation
3. Inspect app logs for error messages

---

**Happy anomaly detecting! 🎉**
