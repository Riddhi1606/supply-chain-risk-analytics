# End-to-End Supply Chain Optimization & Risk Forecasting
## CP42DS Capstone | Evoastra Ventures 

---

## Project Overview

An industrial-grade data science + AI/ML capstone built on the **DataCoSupplyChain dataset** (180,519 records, 53 features). The project spans 4 progressive phases — from raw data normalization through a production-ready risk prediction API and analytics dashboard.

---

## Repository Structure

```
capstone/
├── notebooks/
│   ├── data_cleaning.ipynb              # Phase 1 — Normalization & Cleaning
│   ├── phase2_predictive_modeling.ipynb # Phase 2 — Feature Eng. + Regression + Prophet
│   └── phase3_ml_engine.ipynb           # Phase 3 — XGBoost + SHAP + Risk Assessment
│
├── models/
│   ├── final_late_delivery_model.pkl    # Best classifier (XGBoost tuned)
│   ├── best_regression_model.pkl        # Best regression model
│   ├── prophet_demand_model.json        # Prophet demand forecast
│   ├── scaler.pkl                       # StandardScaler for linear models
│   ├── label_encoders.pkl               # Categorical encoders
│   ├── feature_names.pkl                # Ordered feature list
│   └── model_metadata.json              # Model metrics & config
│
├── dashboard/
│   └── app.py                           # Streamlit analytics dashboard
│
├── report/
│   ├── feature_distributions.png
│   ├── correlation_matrix.png
│   ├── regression_model_comparison.png
│   ├── prophet_forecast.png
│   ├── prophet_components.png
│   ├── confusion_roc.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   ├── shap_waterfall.png
│   └── risk_assessment.png
│
├── architecture/
│   └── deployment_architecture.svg
│
├── main.py                              # FastAPI prediction backend
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Dataset

| Metric | Value |
|---|---|
| Source | DataCoSupplyChainDataset.csv |
| Raw Records | 180,519 |
| Features | 53 |
| Unique Orders | 65,752 |
| Unique Customers | 20,652 |
| Unique Products | 118 |
| Unique Stores | 11,835 |

---

## Phase 1 — Data Analytics (Normalization & Cleaning)

**Notebook:** `notebooks/data_cleaning.ipynb`

**What was done:**
- Decomposed flat table (53 cols × 180,519 rows) into 7 normalized tables (3NF)
- Generated synthetic `Store_id` from Latitude + Longitude
- Removed fully duplicated rows
- Corrected data types (datetime columns)
- Standardized all column names
- Dropped 100%-null `Product_description` column
- Identified and resolved dataset inconsistencies (mislabeled order vs customer attributes)

**Schema:** Customers → Orders → Order_items → Products, Stores → Store_departments, Shipping

---

## Phase 2 — Data Science: Predictive Modeling

**Notebook:** `notebooks/phase2_predictive_modeling.ipynb`

### Feature Engineering (13 new features)
| Feature | Description |
|---|---|
| Delivery_delay | Actual − Scheduled shipping days |
| Is_late | Binary flag for delayed orders |
| Revenue_per_item | Order total / item count |
| Profit_amount | Order total × profit ratio |
| Discount_rate | Total discount / order total |
| Order_value_tier | Quartile bucket (Low/Med/High/Premium) |
| Shipping_efficiency | Scheduled / Actual days |
| Order_month/quarter/dayofweek | Temporal decomposition |
| Is_weekend_order | Binary flag |

### Statistical Testing
| Test | Finding |
|---|---|
| T-Test (order value: late vs on-time) | Significant (p < 0.05) |
| Chi-Square (shipping mode vs late risk) | Significant (p < 0.05) |
| One-Way ANOVA (order total vs market) | Significant (p < 0.05) |

### Regression — Predicting Days_for_Shipping

| Model | MAE | RMSE | R² | Improvement |
|---|---|---|---|---|
| Baseline (mean) | ~1.6 | ~1.8 | 0.00 | — |
| Linear Regression | ~1.3 | ~1.5 | 0.15 | ~19% |
| Ridge | ~1.3 | ~1.5 | 0.15 | ~19% |
| Lasso | ~1.3 | ~1.5 | 0.14 | ~18% |
| Decision Tree | ~0.9 | ~1.1 | 0.48 | ~44% |
| Random Forest | ~0.7 | ~0.9 | 0.68 | ~56% |
| Gradient Boosting | ~0.7 | ~0.9 | 0.70 | ~56% |

✅ All tree models exceed 15% improvement requirement.

### Time Series — Prophet Demand Forecast

| Metric | Value |
|---|---|
| Model | Prophet (multiplicative seasonality) |
| Forecast horizon | 12 weeks |
| MAE | ~52.2 |
| MAPE | < 10% |

---

## Phase 3 — AI/ML Engine

**Notebook:** `notebooks/phase3_ml_engine.ipynb`

### Classification — Late Delivery Risk Prediction

**Target:** `Late_delivery_risk` (binary 0/1)

| Model | Test Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 59.4% | 0.623 |
| Decision Tree | 71.4% | 0.720 |
| Random Forest (base) | 75.2% | 0.801 |
| XGBoost (base) | 76.1% | 0.831 |
| **XGBoost (tuned)** | **76.6%** | **0.847** |

**Cross-Validation (5-Fold Stratified):** AUC = 0.841 ± 0.009

### Hyperparameter Tuning
- RandomizedSearchCV (30 iterations, 5-fold CV)
- Optimized: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, reg_alpha

### SHAP Explainability

**Top 5 Most Influential Features:**
1. `Days_for_shipment` — 31.2% importance (tight window = high risk)
2. `Shipping_mode` — 19.8% (mode strongly predicts lateness)
3. `Delivery_status` — 14.2% (current status feeds back into risk)
4. `Order_total` — 8.9% (higher value orders often expedited differently)
5. `Market` — 7.2% (regional logistics variability)

### Business Risk Assessment

| Risk Tier | Threshold | Expected Late Rate | Action |
|---|---|---|---|
| Low | Risk < 30% | ~8% | Monitor only |
| Medium | 30–60% | ~52% | Flag for review |
| High | > 60% | ~91% | Immediate intervention |

---

## Phase 4 — Deployment-Ready Application

### FastAPI Backend (`main.py`)

**Endpoints:**

| Method | Route | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |
| POST | `/predict` | Single order risk prediction |
| POST | `/predict/batch` | Batch prediction (up to 1000 orders) |
| GET | `/schema/options` | Valid categorical values |

**Interactive docs:** `http://localhost:8000/docs`

### Streamlit Dashboard (`dashboard/app.py`)

5-page analytics suite:
- **Overview Dashboard** — KPI cards, revenue trend, late delivery by mode, market breakdown
- **Risk Predictor** — Interactive form → real-time risk gauge
- **Batch Analysis** — Upload CSV → risk tier segmentation + scatter plot
- **Demand Forecast** — Prophet forecast visualization (12-week horizon)
- **Model Performance** — CV results, feature importance, confusion matrix

---

## Setup Instructions

### Option A: Local (Conda/pip)

```bash
# 1. Clone and navigate
git clone <repo>
cd capstone

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Phase 1 + 2 + 3 notebooks
jupyter notebook notebooks/

# 4. Start API
uvicorn main:app --reload --port 8000

# 5. Start dashboard (separate terminal)
cd dashboard
streamlit run app.py
```

### Option B: Docker (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Services:
#   FastAPI  → http://localhost:8000
#   Swagger  → http://localhost:8000/docs
#   Streamlit → http://localhost:8501
```

### API Usage Example

```python
import requests

payload = {
    "days_for_shipment": 4,
    "order_total": 354.72,
    "total_items": 3,
    "avg_profit_ratio": 0.35,
    "total_discount": 15.0,
    "order_month": 8,
    "order_quarter": 3,
    "order_dayofweek": 2,
    "is_weekend_order": 0,
    "shipping_mode": "Standard Class",
    "market": "Europe",
    "payment_method": "DEBIT",
    "delivery_status": "Shipping on time"
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
# → {'late_delivery_risk': 0, 'risk_probability': 0.27, 'risk_tier': 'Low Risk', ...}
```

---

## Evaluation Summary

| Category | Target | Achieved |
|---|---|---|
| Data Analytics Quality | Complete normalization, EDA | ✅ 7-table schema, 13 engineered features |
| Model Accuracy | >15% over baseline | ✅ 56% improvement (Random Forest), AUC 0.847 |
| ML Implementation | RF + XGBoost + CV + SHAP | ✅ Full pipeline with explainability |
| Business Insight | Risk assessment | ✅ 3-tier risk segmentation + revenue at risk |
| Documentation | Notebooks + README + reports | ✅ Inline markdown + PDF report + this README |
| Code Quality | Clean, modular | ✅ Typed schemas, docstrings, error handling |
| Innovation | Optional app | ✅ FastAPI + Streamlit + Docker (Phase 4 bonus) |

