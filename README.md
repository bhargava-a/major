# AI-Augmented Observability System for ISP

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Performance Metrics](#performance-metrics)
- [API Documentation](#api-documentation)
- [Dashboards](#dashboards)
- [Methodology](#methodology)
- [Target Users](#target-users)
- [Contributing](#contributing)
- [Team](#team)
- [References](#references)
- [License](#license)

---

## 🎯 Overview

The **AI-Augmented Observability System for ISP** is an intelligent, scalable solution designed to detect network anomalies in large-scale Internet Service Provider (ISP), university, and research environments. By leveraging machine learning models (RNN & Transformer-based architectures) integrated with industry-standard observability tools, this system provides real-time anomaly detection, visualization, and alerting capabilities.

### Key Highlights

- **Real-time Anomaly Detection**: Uses RNN and lightweight Transformer models for high-accuracy anomaly scoring
- **Scalable Architecture**: Designed to handle ISP-scale network traffic (from research datasets to production environments)
- **Production-Ready Stack**: Prometheus for metrics storage, FastAPI for inference, Grafana for visualization
- **Enterprise Integration**: REST API for seamless integration with external systems and automation platforms
- **Synthetic Traffic Generation**: Transformer-based traffic generation for testing and validation

---

## ⚠️ Problem Statement

Modern networks have evolved into complex, fast-moving ecosystems with millions of devices connecting daily. Traditional rule-based monitoring systems are inadequate for detecting anomalies in this dynamic environment due to:

### Challenges Addressed

1. **Sudden Traffic Spikes**: Unpredictable, rapid changes in network behavior
2. **Evolving Cyber-Attacks**: DDoS attacks, scanning spikes, and unusual traffic bursts
3. **Unexpected Routing Issues**: Misconfigurations and path failures
4. **New Usage Patterns**: IoT devices and cloud-based services creating variable loads
5. **Manual Monitoring Inefficiency**: Network teams cannot manually inspect massive data volumes

### Impact Scope

- **Students & Faculty**: Slow internet during online classes and lab sessions
- **Research Teams**: Failed large-scale data transfers and interrupted experiments
- **Network Admins**: Hours spent on manual troubleshooting
- **ISPs**: Service degradation and SLA violations
- **Security Teams**: Delayed threat detection and response

---

## ✨ Key Features

### 🤖 Advanced Machine Learning

| Feature | Description |
|---------|-------------|
| **RNN-Based Detection** | Recurrent Neural Networks for sequential anomaly detection |
| **Transformer Models** | Lightweight transformer architecture for pattern recognition |
| **Synthetic Data Generation** | GPT-2 small / Temporal Transformer for realistic traffic simulation |
| **Hybrid Approach** | Combines multiple models for improved accuracy and robustness |

### 📊 Observability & Visualization

- Real-time Prometheus metrics collection and storage
- Dynamic Grafana dashboards with custom panels
- Traffic pattern visualization and trend analysis
- Automated alert generation for critical anomalies

### 🔌 Integration & Automation

- FastAPI-based REST API for model inference
- JSON request/response format for easy integration
- Webhook support for external alert systems
- CI/CD ready architecture

### 📈 Data Processing

- Efficient data pruning: ~40GB → ~4GB usable dataset
- Feature extraction and normalization
- Time-series windowing and batching
- Real-time metric aggregation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  CESNET Dataset (Raw Traffic)           │
├─────────────────────────────────────────────────────────┤
│                  Data Preprocessing & Pruning            │
├─────────────────────────────────────────────────────────┤
│              Feature Extraction & Normalization          │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌──────────────────────┐  │
│  │  RNN Detector    │         │ Transformer Generator│  │
│  │ (Anomaly Score)  │         │ (Synthetic Traffic)  │  │
│  └──────────────────┘         └──────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│            FastAPI Inference Service                     │
├─────────────────────────────────────────────────────────┤
│         Prometheus Metrics Storage                       │
├─────────────────────────────────────────────────────────┤
│         Grafana Visualization & Dashboards              │
├─────────────────────────────────────────────────────────┤
│  REST API  │  Webhooks  │  Alert System  │  External Apps│
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Core Technologies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Language** | Python | 3.8+ | Primary development language |
| **ML Framework** | PyTorch | 2.0+ | Deep learning model implementation |
| **ML Library** | Scikit-Learn | 1.0+ | Data preprocessing & evaluation |
| **Metrics Storage** | Prometheus | Latest | Time-series metrics database |
| **Visualization** | Grafana | Latest | Interactive dashboards |
| **API Framework** | FastAPI | 0.95+ | REST API service |
| **Data Processing** | Pandas, NumPy | Latest | Data manipulation & analysis |

### Supporting Tools

- **Jupyter Notebooks**: Experiment tracking and prototyping
- **Docker**: Containerization and deployment
- **Git**: Version control
- **Linux/Windows**: Cross-platform support

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- 8GB+ RAM (recommended for dataset processing)
- GPU support (NVIDIA CUDA) optional but recommended

### Step 1: Clone Repository

```bash
git clone https://github.com/bhargava-a/major.git
cd major
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n observability python=3.8
conda activate observability
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Setup Configuration

```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your settings
```

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
# Data Configuration
dataset:
  path: "/path/to/cesnet/data"
  pruning_ratio: 0.1
  validation_split: 0.2

# Model Configuration
models:
  rnn:
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
  transformer:
    num_heads: 8
    num_layers: 4
    hidden_dim: 256

# Prometheus Configuration
prometheus:
  host: "localhost"
  port: 9090
  scrape_interval: 15s

# FastAPI Configuration
api:
  host: "0.0.0.0"
  port: 8000
  debug: false

# Alerting
alerts:
  anomaly_threshold: 0.75
  webhook_url: "https://your-webhook-url"
```

---

## 🚀 Usage

### Data Preprocessing

```bash
python csv_to_npz.py --input data/raw/ --output data/processed/
```

### Model Training

```bash
python train_rnn.py --config config/config.yaml --epochs 100
python train_transformer.py --config config/config.yaml --epochs 50
```

### Data Merging & Aggregation

```bash
python merging.py --input data/processed/ --output data/merged/
python npz_merging.py --files data/merged/*.npz
```

### Start FastAPI Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Launch Prometheus & Grafana

```bash
docker-compose up -d prometheus grafana
```

---

## 📁 Project Structure

```
major/
├── data/
│   ├── raw/                    # Original CESNET dataset
│   ├── processed/              # Preprocessed data
│   └── merged/                 # Merged NPZ files
├── models/
│   ├── rnn_detector.py         # RNN-based anomaly detector
│   ├── transformer_model.py    # Transformer-based models
│   └── checkpoints/            # Saved model weights
├── api/
│   ├── main.py                 # FastAPI application
│   ├── routes/                 # API endpoints
│   └── schemas/                # Request/response schemas
├── notebooks/
│   ├── data_exploration.ipynb  # EDA and analysis
│   └── model_validation.ipynb  # Model testing
├── config/
│   ├── config.yaml             # Main configuration
│   └── prometheus.yml          # Prometheus config
├── dashboards/
│   └── grafana/                # Grafana dashboard definitions
├── scripts/
│   ├── csv_to_npz.py           # CSV to NPZ conversion
│   ├── merging.py              # Data merging utility
│   └── npz_merging.py          # NPZ merging utility
├── tests/
│   └── unit_tests/             # Test cases
├── docker-compose.yml          # Docker services configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 📊 Dataset

### CESNET Time Series 24 Dataset

- **Source**: Czech academic network (CESNET)
- **Size**: ~40GB (raw) → ~4GB (processed)
- **Format**: NetFlow records with timestamps
- **Time Period**: Multiple years of network traffic
- **Features**: Source/destination IPs, ports, bytes transferred, protocols, flow counts

### Data Processing Pipeline

1. **Extraction**: Parse NetFlow records and timestamps
2. **Cleaning**: Remove incomplete or malformed records
3. **Pruning**: Filter noise and irrelevant traffic (~90% reduction)
4. **Feature Engineering**: Extract relevant metrics (bytes/sec, flows/sec, entropy, etc.)
5. **Normalization**: Scale features to [0, 1] range
6. **Windowing**: Create time-series windows for model training

---

## 🤖 Models

### RNN-Based Anomaly Detector

**Architecture**: Multi-layer LSTM with attention mechanism

```
Input (Time-series window)
    ↓
Embedding Layer
    ↓
LSTM Layer 1 (hidden_size=128)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (hidden_size=128)
    ↓
Attention Mechanism
    ↓
Fully Connected Layer
    ↓
Output (Anomaly Score: 0-1)
```

**Hyperparameters**:
- Hidden Size: 128
- Number of Layers: 2
- Dropout: 0.2
- Learning Rate: 0.001
- Batch Size: 32

### Transformer-Based Generator

**Purpose**: Generate synthetic realistic network traffic for validation

**Architecture**: Lightweight transformer with encoder-decoder attention

**Capabilities**:
- Learns traffic patterns from historical data
- Generates time-correlated synthetic traffic
- Supports conditional generation (traffic type, peak hours)

---

## 📈 Performance Metrics

### Evaluation Metrics

| Metric | Target | Status |
|--------|--------|--------|
| **Precision** | > 95% | In Development |
| **Recall** | > 90% | In Development |
| **F1-Score** | > 92% | In Development |
| **Inference Latency** | < 100ms | Target |
| **Memory Footprint** | < 2GB | Target |
| **Throughput** | > 10,000 samples/sec | Target |

### Benchmark Results

- **Training Time**: ~2-4 hours on GPU (RTX 3080+)
- **Model Size**: RNN (~50MB), Transformer (~80MB)
- **False Positive Rate**: Target < 5%
- **Detection Coverage**: > 85% of known anomalies

---

## 🔌 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

#### 2. Anomaly Detection
```http
POST /api/v1/detect
```

**Request Body**:
```json
{
  "metrics": {
    "bytes_per_sec": 1250.5,
    "flows_per_sec": 85.3,
    "entropy": 6.2,
    "packet_count": 450,
    "timestamp": "2025-12-14T10:30:00Z"
  }
}
```

**Response**:
```json
{
  "anomaly_score": 0.78,
  "is_anomaly": true,
  "confidence": 0.92,
  "model": "rnn",
  "timestamp": "2025-12-14T10:30:00Z"
}
```

#### 3. Batch Detection
```http
POST /api/v1/detect-batch
```

**Request Body**:
```json
{
  "metrics": [
    { /* metrics object 1 */ },
    { /* metrics object 2 */ }
  ]
}
```

#### 4. Traffic Synthesis
```http
POST /api/v1/generate-traffic
```

**Request Body**:
```json
{
  "num_samples": 100,
  "traffic_type": "normal",
  "duration_sec": 300
}
```

---

## 📊 Dashboards

### Grafana Dashboard Features

#### Panel 1: Real-Time Traffic Overview
- Current bytes/sec and flows/sec
- Active connections count
- Protocol distribution pie chart

#### Panel 2: Anomaly Timeline
- Anomaly scores over time
- Critical anomaly markers
- Anomaly type classification

#### Panel 3: Model Performance
- Inference latency trend
- Model confidence scores
- False positive rate

#### Panel 4: Alerts & Events
- Recent critical anomalies
- Alert history
- Attack pattern timeline

#### Panel 5: System Health
- API response times
- Prometheus storage usage
- Model inference queue length

---

## 🔬 Methodology

### Phase 1: Literature Review
- Study ML-based anomaly detection techniques
- Research RNNs, LSTMs, and Transformer architectures
- Analyze existing ISP-scale datasets and benchmarks

### Phase 2: Dataset Exploration & Preprocessing
- Parse CESNET NetFlow records
- Perform statistical analysis and anomaly labeling
- Implement pruning strategies to reduce dataset size
- Extract and engineer relevant features

### Phase 3: Model Development & Comparison
- Build baseline RNN anomaly detector
- Implement Transformer-based traffic generator
- Compare accuracy, precision, recall, and latency
- Optimize models for production constraints

### Phase 4: Integration & Pipeline Development
- Develop FastAPI inference service
- Configure Prometheus metrics collection
- Build automated data ingestion pipeline
- Implement alert triggering mechanism

### Phase 5: Visualization & Testing
- Design Grafana dashboards
- Create alerting rules
- Perform end-to-end system testing
- Validate on both synthetic and real data

---

## 👥 Target Users

### Internet Service Providers (ISPs)
- Monitor large-scale network traffic in real-time
- Detect DDoS spikes, bandwidth anomalies, and routing issues
- Reduce downtime and maintain SLA compliance
- Lower operational costs through automation

### University & Research Networks
- Ensure stable connectivity for students, faculty, and researchers
- Detect traffic surges during peak academic hours
- Protect critical research data from cyber threats
- Enable seamless collaboration and data transfers

### Network Security Teams (SOC)
- Replace manual log inspection with automated insights
- Respond faster to potential attacks and misconfigurations
- Correlate multiple data sources for threat detection
- Generate actionable intelligence from network data

### IT Infrastructure & Network Admins
- Access real-time dashboards for traffic trends
- Receive automated alerts for potential issues
- Reduce mean-time-to-detection (MTTD)
- Enable proactive network management

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👨‍💼 Team

**Project Leads & Contributors**:

| Name | ID | Role |
|------|----|----|
| Bhargava A | R22EK003 | Lead Developer |
| Basavaraj R Bagewadi | R22EK030 | ML Engineer |
| Krithika S | R22ER072 | Data Engineer |
| M K Varun Gowda | R22EK040 | Backend Developer |

**Supervisor**: Prof. Kavita Babalad

**Institution**: School of Computing and Information Technology, REVA University, Bengaluru

---

## 📚 References

### Research Papers & Datasets

1. **CESNET Time Series 24 Dataset** – High-speed academic network traffic logs for anomaly detection research

2. Lakhina, A., Crovella, M., & Diot, C. (2004). "Mining anomalies using traffic feature distributions." *ACM SIGCOMM*.

3. Chandola, V., Banerjee, A., & Kumar, V. (2009). "Anomaly detection: A survey." *ACM Computing Surveys*.

4. Xu, K., Zhang, Z., & Bhattacharyya, S. (2008). "Internet traffic behavior profiling for network security monitoring." *IEEE/ACM Transactions on Networking*.

5. Ahmed, E., et al. "A survey on network anomaly detection using machine learning." *Journal of Network & Computer Applications*.

6. Erfani, S. M., et al. "High-dimensional anomaly detection using deep learning." *IEEE Transactions on Information Forensics and Security*.

7. Breunig, M. M., et al. "LOF: Local Outlier Factor." *ACM SIGMOD Conference Proceedings*.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Support & Contact

For issues, questions, or suggestions, please:

- Open an issue on GitHub
- Contact the development team at your-email@example.com
- Check the project wiki for additional documentation

---

**Last Updated**: December 14, 2025

**Status**: Active Development - Phase 1 Complete, Phase 2-5 In Progress

---