# MP190 — Tomorrow's Air Quality (Kochi)
## SCSE3040 MLOps Mini Project — Group G209

---

### Project Metadata & Team Identification

- **Group ID**: G209
- **Problem ID**: MP190
- **Faculty Mentor**: Saurabh Shrivastava
- **Batch**: 53

#### Group Members & Roles

| Enrolment No. | Student Name | Role | Core Ownership |
|---|---|---|---|
| **S24CSEU1065** | Sujal Ranga | Ops Lead | GitHub Actions CI, Kubernetes manifests, structured JSON logging, drift (PSI) & retraining |
| **S24CSEU1058** | Poojita Alla Chowdary | Data Lead | DVC tracking, seven data-quality audits, cleaning pipeline, `findings.yaml` data part |
| **S24CSEU1045** | Aditya Bhatt | Service Lead | FastAPI inference microservice, Pydantic validation rules, Pytest suite, Dockerfile |
| **S24CSEU1038** | Abhinav Bhardwaj | Model Lead | Baseline modeling, MLflow experiment tracking & registry, holdout predictions |

---

## 1. Executive Summary & Client Context

### Client
**Kochi Clean Air Cell** (India)

### Context & Mission
The Kochi Clean Air Cell operates ambient air-quality monitoring stations across diverse geographical zones in Kochi (Central, Residential, Industrial). Every evening, educational institutions, schools, and healthcare facilities inquire whether tomorrow will be safe for outdoor athletic and extracurricular activities. A reliable 24-hour advance forecast enables schools to reschedule outdoor sports indoors and hospitals to proactively mobilize respiratory and asthma care resources.

### Objective
Develop, track, containerize, test, and deploy an end-to-end, production-grade MLOps regression pipeline predicting tomorrow's ambient air quality:
- **Target Variable**: `pm25_next_day` — Average PM2.5 level on the following day ($\mu g/m^3$).
- **Task Type**: Supervised Regression.
- **Primary Evaluation Metric**: Mean Absolute Error (MAE) in $\mu g/m^3$.

---

## 2. Dataset Overview

The project operates on three sequential datasets provided by Kochi Clean Air Cell:

| Dataset | Relative Path | Rows | Period | Description |
|---|---|---|---|---|
| **Training Data** | `data/raw/train.csv` | ~1,000 | Jan – Jun 2026 | Messy historical station records exhibiting real-world data issues. |
| **Production Feed** | `data/raw/later.csv` | 300 | Jul – Aug 2026 | Clean operational data arriving after model deployment for monitoring. |
| **Holdout Evaluation** | `data/raw/holdout_features.csv` | 250 | Sep 2026 | Unlabeled evaluation records for final benchmark submission. |

### Feature Dictionary

| Column | Type | Unit | Description | In Inference API? |
|---|---|---|---|---|
| `record_id` | Text | — | Unique identifier of station record | No (metadata only) |
| `date` | Date | YYYY-MM-DD | Date of observation | No (used for temporal split) |
| `pm25_today` | Numeric | $\mu g/m^3$ | Average measured PM2.5 today | **Yes** |
| `wind_kmph` | Numeric | $km/h$ | Average wind speed | **Yes** (Must be $\ge 0$) |
| `humidity_pct` | Numeric | $\%$ | Relative humidity percentage | **Yes** |
| `temp_c` | Numeric | $^{\circ}C$ | Average ambient temperature | **Yes** (Must be $\le 70$) |
| `traffic_index` | Numeric | Index | Traffic volume count index | **Yes** |
| `stubble_fires` | Numeric | Count | Satellite fire detections within 200 km | **Yes** |
| `day_type` | Text | Category | Type of day: `weekday`, `weekend`, `festival` | **Yes** |
| `station_zone` | Text | Category | Location category: `Central`, `Residential`, `Industrial` | **Yes** |
| `pm25_measured_next_day` | Numeric | $\mu g/m^3$ | Actual next-day measurement | **No** (Target leakage!) |
| `pm25_next_day` | Numeric | $\mu g/m^3$ | **Target variable** to predict | Target (Training only) |

---

## 3. Seven Data-Quality Problem Framework

Per Student Handbook Section 7, `data/raw/train.csv` is audited for seven distinct real-world data anomalies:

1. **`leaking_column`**: Detection of future-dated information (e.g., `pm25_measured_next_day`) absent from `holdout_features.csv` that falsely inflates training metrics.
2. **`unit_mixup`**: Identification of specific station zones where sensors reported values in different metric scales (e.g., metres vs. kilometers, factor 1000).
3. **`sentinel_value`**: Identification of numeric placeholder codes (e.g. `-999`, `-1`, `9999`) mistakenly entered in lieu of missing values.
4. **`duplicate_rows`**: Counting and pruning of exact redundant row duplicates.
5. **`stuck_sensor`**: Identification of freeze-failure periods where a sensor reported identical readings across contiguous calendar dates.
6. **`target_typos`**: Identification of keystroke decimal errors (e.g., targets typed 10x too large).
7. **`spelling_variants`**: Detection and normalization of casing, whitespace, or typographical inconsistencies in categorical columns compared against `later.csv`.

All findings are formally tracked in `findings.yaml` with reproducible detection algorithms maintained in `src/data_validation.py`.

---

## 4. Repository Structure

```
G209_MP190/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI test suite running on push
├── app/
│   ├── __init__.py              # FastAPI package definition
│   ├── main.py                  # API endpoints (/health, /predict) & JSON logger
│   └── schemas.py               # Pydantic request/response models with validation rules
├── data/
│   ├── raw/
│   │   ├── train.csv            # Original training data (preserved)
│   │   ├── later.csv            # Production monitoring data
│   │   └── holdout_features.csv # Final evaluation feature set
│   └── processed/
│       └── .gitkeep             # Destination for cleaned datasets
├── k8s/
│   ├── deployment.yaml          # Kubernetes Deployment (mp190-api, 4 replicas)
│   └── service.yaml             # Kubernetes Service (mp190-svc, port 80 -> 8000)
├── models/
│   └── .gitkeep                 # Saved model artifacts (model.joblib)
├── predictions/
│   └── .gitkeep                 # Final predictions (holdout_predictions.csv)
├── src/
│   ├── __init__.py              # Python package marker
│   ├── config.py                # Configuration loader reading params.yaml
│   ├── logging_config.py        # Structured JSON formatter writing to stdout
│   ├── data_validation.py       # Seven data-quality detection routines
│   ├── data_cleaning.py         # Data sanitization and remediation pipeline
│   ├── feature_engineering.py   # ColumnTransformer and date feature engineering
│   ├── train.py                 # MLflow tracking, SHA-256 logging, model training
│   ├── evaluate.py              # MAE, RMSE, R2, and station-zone evaluations
│   ├── predict.py               # Batch inference runner for holdout features
│   ├── drift.py                 # 10-bin Population Stability Index (PSI) calculator
│   └── retrain.py               # v2 chronological retraining and 10% promotion rule
├── tests/
│   ├── __init__.py              # Test package marker
│   ├── test_api.py              # TestClient integration tests (/health, /predict, 422)
│   ├── test_data_cleaning.py    # Unit tests for cleaning transformations
│   ├── test_model.py            # Unit tests for evaluation metrics & promotion logic
│   └── test_validation.py       # Unit tests for Pydantic boundary validation
├── .dockerignore                # Container build exclusions
├── .gitignore                   # Git exclusions (preserves mlflow.db and joblib)
├── Dockerfile                   # python:3.12-slim production container manifest
├── findings.yaml                # Assessment audit schema
├── GROUP.md                     # Official group allocation metadata
├── params.yaml                  # Centralized, configurable pipeline parameters
├── PROBLEM.md                   # Problem MP190 specification
├── README.md                    # Comprehensive project documentation
├── requirements.txt             # Pinned dependencies (==)
└── STUDENT_HANDBOOK.md          # Course syllabus, rules, and marking criteria
```

---

## 5. Configuration Management (`params.yaml`)

All parameters (paths, feature lists, hyperparameters, monitoring thresholds, service ports) are isolated in `params.yaml` and loaded via `src/config.py` using `yaml.safe_load`:

```yaml
project:
  group_id: "G209"
  problem_id: "MP190"
  client: "Kochi Clean Air Cell"
  target_column: "pm25_next_day"
  primary_metric: "mae"

paths:
  raw_train: "data/raw/train.csv"
  raw_later: "data/raw/later.csv"
  raw_holdout: "data/raw/holdout_features.csv"
  model_path: "models/model.joblib"
  holdout_predictions: "predictions/holdout_predictions.csv"
  findings_path: "findings.yaml"
  mlflow_tracking_uri: "sqlite:///mlflow.db"

monitoring:
  psi_bins: 10
  retrain_split_ratio: 0.70
  promotion_threshold: 0.10
```

---

## 6. Machine Learning Lifecycle & MLflow

### Experiment Tracking Setup
- **Tracking URI**: SQLite database located in repo root: `sqlite:///mlflow.db`
- **Data Integrity Hash**: Every experiment run computes and logs `data_sha256`:
  $$\text{SHA-256}(\text{train.csv}) = \texttt{f2cfe6bd7cf82f8f07b545547abd9e36a15510b6a4cd44d78fa4a539b3d496db}$$
- **Primary Metric**: Logged under `mae` or `val_mae`.
- **Model Registry**: Champion model tagged with the `@champion` alias:
  ```python
  from mlflow import MlflowClient
  client = MlflowClient()
  client.set_registered_model_alias(name="mp190-pm25-predictor", alias="champion", version="1")
  ```

---

## 7. Inference Microservice (FastAPI)

### Specification
- Framework: `FastAPI` + `Uvicorn`
- Base Path: `0.0.0.0:8000`

### Endpoints

#### 1. `GET /health`
Returns service operational status.
```json
{
  "status": "ok",
  "service": "mp190-api"
}
```

#### 2. `POST /predict`
Accepts 8 observation features and returns predicted PM2.5.
```json
{
  "pm25_today": 45.2,
  "wind_kmph": 12.5,
  "humidity_pct": 78.0,
  "temp_c": 29.5,
  "traffic_index": 120.0,
  "stubble_fires": 3.0,
  "day_type": "weekday",
  "station_zone": "Central"
}
```

**Response (HTTP 200)**:
```json
{
  "prediction": 48.94,
  "status": "success",
  "model_version": "1.0"
}
```

### Client Validation Rules
Enforced in `app/schemas.py` using Pydantic `Field` constraints. Invalid requests trigger an immediate **HTTP 422 Unprocessable Entity**:
1. `temp_c > 70` is impossible in Kochi; rejected with `Field(..., le=70.0)`.
2. `wind_kmph < 0` is physically impossible; rejected with `Field(..., ge=0.0)`.

### Structured JSON Logging
Predictions emit single-line structured JSON logs directly to `sys.stdout`:
```json
{"timestamp": "2026-11-18T09:00:00Z", "level": "INFO", "logger": "mp190-api", "message": "Prediction generated", "event": "prediction", "model_version": "1.0", "prediction": 48.94, "station_zone": "Central", "day_type": "weekday", "status": "success"}
```

---

## 8. Containerization & Kubernetes

### Docker Specifications
- **Image**: `mp190-api:1.0`
- **Base Image**: `python:3.12-slim`
- **Port**: `8000`
- **Build Command**:
  ```bash
  docker build -t mp190-api:1.0 .
  ```
- **Run Command**:
  ```bash
  docker run -d -p 8000:8000 --name mp190-service mp190-api:1.0
  ```

### Kubernetes Specifications
- **Deployment**: `k8s/deployment.yaml`
  - Name: `mp190-api`
  - Replicas: `4`
  - Image: `mp190-api:1.0`
  - Probes: Readiness (`/health`, 5s initial delay) & Liveness (`/health`, 15s initial delay)
- **Service**: `k8s/service.yaml`
  - Name: `mp190-svc`
  - Port: `80` forwarding to targetPort `8000`
  - Selector: `app: mp190-api`

---

## 9. Monitoring, Retraining & Promotion

1. **Covariate Drift (PSI)**:
   $$\text{PSI} = \sum \left( \% \text{Actual} - \% \text{Expected} \right) \times \ln\left(\frac{\% \text{Actual}}{\% \text{Expected}}\right)$$
   Calculated across 10 quantile bins for each numeric column between `train.csv` and `later.csv`.
2. **Subgroup Performance**:
   Evaluate v1 MAE independently across `station_zone` categories (`Central`, `Residential`, `Industrial`) to identify the worst-served subgroup.
3. **Model Retraining (v2)**:
   Chronologically sort `later.csv` by date. Retrain v2 on `train.csv` plus the first 70% of `later.csv`.
4. **10% Promotion Rule**:
   $$\text{Promote } v2 \iff \text{MAE}_{v2}(\text{last 30\%}) \le 0.90 \times \text{MAE}_{v1}(\text{last 30\%})$$
   If promoted, v2 receives the `@champion` alias in the MLflow model registry.

---

## 10. Automated Testing & Continuous Integration

### Test Suite Structure
The Pytest suite in `tests/` executes self-contained unit and integration tests:
- `tests/test_api.py`: FastAPI `TestClient` tests verifying `/health`, `/predict`, and HTTP 422 validation.
- `tests/test_data_cleaning.py`: Unit tests for deduplication, leakage removal, and sentinel imputation.
- `tests/test_model.py`: Unit tests for MAE, RMSE, R2, and retraining promotion logic.
- `tests/test_validation.py`: Pydantic schema boundary checks.

Run tests locally:
```bash
pytest tests/ -v
```

### GitHub Actions CI Workflow
Defined in `.github/workflows/ci.yml`:
- Begins with the mandatory `on:` trigger for `push` events.
- Sets up Python 3.12, installs pinned `requirements.txt`, and executes `pytest tests/ -v`.
- Requires zero external CSV files, guaranteeing green CI runs on GitHub.

---

## 11. Final Prediction Generation

When the champion model is locked, final predictions for `data/raw/holdout_features.csv` (250 rows) are generated via `src/predict.py` and saved to `predictions/holdout_predictions.csv`:
```csv
record_id,prediction
REC_001,42.50
REC_002,38.10
...
```

---

## 12. Reproduction Guide

```bash
# 1. Clone repository
git clone <repo-url>
cd G209_MP190

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Run test suite
pytest tests/ -v

# 5. Launch FastAPI local server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
