## Customer Segmentation & Prediction Pipeline (RFM + KMeans + PCA + GPR)

# Situation
Retail transaction data is often noisy and fragmented, making it difficult to identify valuable customers, churn risk, and revenue opportunities.

# Task
Build a reproducible pipeline that:

- segments customers,

- flags churn risk, and

- predicts customer monetary value.

# Action
This project implements an end-to-end workflow using Python:

- Data validation & cleaning: removes duplicates, drops missing CustomerID rows, filters refunds/negative spend

- Feature engineering: creates RFM metrics (Recency, Frequency, Monetary)

- Scaling: StandardScaler for modelling inputs (saved to disk)

- Segmentation: KMeans clustering with k=2..10, evaluated using inertia + silhouette (metrics exported)

- Explainability: cluster profiling + human-readable labels (High Value, Loyal, Regular, Churn Risk)

- Churn logic: Recency thresholds using Tukey fence + Q3 comparison; exports churned vs active lists

- Visual analytics: distribution plots (IQR/extremes), correlation heatmap, cluster size chart

- Dimensionality reduction: PCA 2D & 3D for segment visualisation

- Prediction: Gaussian Process Regression to predict Monetary value and uncertainty, plus diagnostics (actual vs predicted, residuals)

- Reproducibility: timestamped output folders + automatic archiving; model artefacts persisted with joblib

# Outputs
Each run generates a timestamped folder under:
outputs/outputs_YYYYMMDD_HHMM/
including:

- customers_segmented_master.csv (Tableau/Power BI ready)

- customers_segmented_master.xlsx

- Cluster profiles + KMeans metrics

- Per-segment exports (marketing automation ready)

- Saved models: scaler, KMeans, GPR

- PNG plots (dpi=200)

# How to run

Place dataset in: data/Online Retail.xlsx

Run: python src/customer_segmentation_master.py

Review outputs in the timestamped folder under outputs/

# Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib/seaborn, joblib

# About

Built a full customer segmentation pipeline using RFM + KMeans, created marketing-friendly segments (Champions, At-Risk, Loyal), and exported BI-ready datasets and plots for stakeholders

Trained a regression model to predict customer monetary value and uncertainty, which is useful for prioritising retention offers and forecasting revenue contribution by segment.
