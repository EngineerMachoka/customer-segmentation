import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import norm
import joblib

# === Define paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# === Filenames ===
DATA_FILE = 'Online Retail.xlsx'
file_path = os.path.join(DATA_DIR, DATA_FILE)

# Output files
elbow_path = os.path.join(OUTPUT_DIR, 'elbow_plot_v2.png')
silhouette_path = os.path.join(OUTPUT_DIR, 'silhouette_plot_v2.png')
pca_variance_path = os.path.join(OUTPUT_DIR, 'pca_variance_v2.png')
rfm_plot_path = os.path.join(OUTPUT_DIR, 'rfm_segmentation_v2.png')
pca_plot_path = os.path.join(OUTPUT_DIR, 'pca_segmentation_v2.png')
csv_output_path = os.path.join(OUTPUT_DIR, 'customers_segmented_v2.csv')
metrics_output_path = os.path.join(OUTPUT_DIR, 'kmeans_metrics_v2.csv')
cluster_profile_path = os.path.join(OUTPUT_DIR, 'cluster_profiles_v2.csv')
scaler_path = os.path.join(OUTPUT_DIR, 'rfm_scaler.pkl')
model_path = os.path.join(OUTPUT_DIR, 'kmeans_model.pkl')

# New output for distribution charts
distribution_dir = os.path.join(OUTPUT_DIR, 'distributions')
os.makedirs(distribution_dir, exist_ok=True)

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Data ===
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

data = pd.read_excel(file_path)
data.drop_duplicates(inplace=True)
data.dropna(subset=['CustomerID'], inplace=True)
data['total_spent'] = data['UnitPrice'] * data['Quantity']
data = data[data['total_spent'] > 0]
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
reference_date = data['InvoiceDate'].max()

# === Create RFM Features ===
rfm = data.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('total_spent', 'sum')
).reset_index()

# === Standardize ===
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# === Elbow + Silhouette ===
inertia = []
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, labels))

metrics_df = pd.DataFrame({'k': list(k_range), 'Inertia': inertia, 'Silhouette': silhouette_scores})
metrics_df.to_csv(metrics_output_path, index=False)

# === Ask user for optimal k or auto-select ===
print(f"ℹ️ Silhouette-optimal k = {np.argmax(silhouette_scores) + 2}")
user_input = input("Enter k (or press Enter to auto-select): ").strip()
if user_input.isdigit():
    optimal_k = int(user_input)
else:
    optimal_k = np.argmax(silhouette_scores) + 2

# === Final KMeans Clustering ===
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Save model and scaler
joblib.dump(scaler, scaler_path)
joblib.dump(kmeans, model_path)

# === PCA for Visualization ===
pca = PCA(n_components=2, random_state=42)
rfm_pca = pca.fit_transform(rfm_scaled)
rfm['PCA1'] = rfm_pca[:, 0]
rfm['PCA2'] = rfm_pca[:, 1]

# === Cluster Summary ===
cluster_summary = rfm.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'}).round(1)

# === Assign Human-Friendly Cluster Names ===
def assign_segment_labels(summary_df):
    labels = {}
    monetary_rank = summary_df['Monetary'].rank(method='min', ascending=False)
    recency_rank = summary_df['Recency'].rank(method='min')
    frequency_rank = summary_df['Frequency'].rank(method='min', ascending=False)
    for idx in summary_df.index:
        if monetary_rank[idx] == 1:
            labels[idx] = 'High Value'
        elif frequency_rank[idx] == 1:
            labels[idx] = 'Loyal'
        elif recency_rank[idx] == summary_df['Recency'].rank().max():
            labels[idx] = 'Churn Risk'
        else:
            labels[idx] = 'Regular'
    return labels

segment_labels = assign_segment_labels(cluster_summary)
rfm['SegmentLabel'] = rfm['Segment'].map(segment_labels)
cluster_summary['Label'] = cluster_summary.index.map(segment_labels)
cluster_summary.to_csv(cluster_profile_path)

# === Distribution Visualizations ===

# Gaussian + IQR Chart
def plot_distribution_with_iqr(data, column):
    plt.figure(figsize=(7, 4))
    sns.histplot(data[column], bins=30, kde=True, color='skyblue', stat='density', edgecolor='black')
    mu, std = norm.fit(data[column])
    x = np.linspace(data[column].min(), data[column].max(), 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', label='Gaussian Fit')
    q1, q3 = np.percentile(data[column], [25, 75])
    plt.axvline(q1, color='green', linestyle='--', label='Q1 (25%)')
    plt.axvline(q3, color='purple', linestyle='--', label='Q3 (75%)')
    plt.title(f'Distribution of {column} with IQR and Gaussian Fit')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(distribution_dir, f"{column}_gaussian_iqr.png"))
    plt.close()

# Bayesian-style KDE Plot
def plot_bayesian_kde(data, column):
    plt.figure(figsize=(7, 4))
    sns.kdeplot(data[column], fill=True, color='navy', bw_adjust=0.5)
    plt.title(f'Bayesian-style KDE for {column}')
    plt.tight_layout()
    plt.savefig(os.path.join(distribution_dir, f"{column}_bayesian_kde.png"))
    plt.close()

# Apply plots to all RFM columns
for col in ['Recency', 'Frequency', 'Monetary']:
    plot_distribution_with_iqr(rfm, col)
    plot_bayesian_kde(rfm, col)

# === Export Clustered Data ===
rfm.to_csv(csv_output_path, index=False)

# === Final Output ===
print("\n📊 Cluster Profiles:\n")
print(cluster_summary[['Label', 'Recency', 'Frequency', 'Monetary', 'Count']])
print(f"\n✅ Segmentation complete. Results saved to: {csv_output_path}")
