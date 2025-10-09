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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from mpl_toolkits.mplot3d import Axes3D

# ============================================
# CONFIGURATION
# ============================================
AUTO_MODE = True  # Run automatically (non-interactive)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

DATA_FILE = 'Online Retail.xlsx'
file_path = os.path.join(DATA_DIR, DATA_FILE)

# Output paths
csv_output_path = os.path.join(OUTPUT_DIR, 'customers_segmented_v2.csv')
xlsx_output_path = os.path.join(OUTPUT_DIR, 'customers_segmented_v2.xlsx')
metrics_output_path = os.path.join(OUTPUT_DIR, 'kmeans_metrics_v2.csv')
cluster_profile_path = os.path.join(OUTPUT_DIR, 'cluster_profiles_v2.csv')
distribution_dir = os.path.join(OUTPUT_DIR, 'distributions')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(distribution_dir, exist_ok=True)

# ============================================
# LOAD & CLEAN DATA
# ============================================
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Data file not found: {file_path}")

print("📦 Loading and preparing data...")
data = pd.read_excel(file_path)
data.drop_duplicates(inplace=True)
data.dropna(subset=['CustomerID'], inplace=True)
data['total_spent'] = data['UnitPrice'] * data['Quantity']
data = data[data['total_spent'] > 0]
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
reference_date = data['InvoiceDate'].max()

# ============================================
# CREATE RFM FEATURES
# ============================================
print("🧮 Generating RFM metrics...")
rfm = data.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('total_spent', 'sum')
).reset_index()

# Standardize
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# ============================================
# FIND OPTIMAL NUMBER OF CLUSTERS
# ============================================
print("🔍 Finding optimal cluster count (KMeans)...")
inertia, silhouette_scores = [], []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, labels))

metrics_df = pd.DataFrame({'k': list(k_range), 'Inertia': inertia, 'Silhouette': silhouette_scores})
metrics_df.to_csv(metrics_output_path, index=False)

optimal_k = np.argmax(silhouette_scores) + 2
print(f"✅ Optimal number of clusters determined: {optimal_k}")

# ============================================
# FINAL CLUSTERING
# ============================================
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'rfm_scaler.pkl'))
joblib.dump(kmeans, os.path.join(OUTPUT_DIR, 'kmeans_model.pkl'))

# ============================================
# PCA FOR VISUALIZATION
# ============================================
pca = PCA(n_components=2, random_state=42)
rfm_pca = pca.fit_transform(rfm_scaled)
rfm['PCA1'], rfm['PCA2'] = rfm_pca[:, 0], rfm_pca[:, 1]

# ============================================
# CLUSTER SUMMARY
# ============================================
cluster_summary = rfm.groupby('Segment').agg({
    'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean', 'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'}).round(1)

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

# ============================================
# DISTRIBUTION PLOTS
# ============================================
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
    plt.title(f'{column} Distribution with Gaussian Fit')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(distribution_dir, f"{column}_distribution.png"), dpi=200)
    plt.close()

for col in ['Recency', 'Frequency', 'Monetary']:
    plot_distribution_with_iqr(rfm, col)

# ============================================
# TABLEAU-READY CLUSTER VISUALIZATION
# ============================================
print("🎨 Generating Tableau-ready segmentation charts...")

segment_palette = {
    'High Value': '#FF6B6B',
    'Loyal': '#4ECDC4',
    'Regular': '#FFD93D',
    'Churn Risk': '#1A535C'
}

# --- RFM Chart ---
plt.figure(figsize=(8,5))
for label, color in segment_palette.items():
    subset = rfm[rfm['SegmentLabel'] == label]
    plt.scatter(subset['Recency'], subset['Monetary'],
                label=label, color=color, alpha=0.7, edgecolor='k', s=60)

centroids_df = pd.DataFrame(centroids, columns=['Recency', 'Frequency', 'Monetary'])
plt.scatter(centroids_df['Recency'], centroids_df['Monetary'],
            color='red', marker='X', s=200, label='Centroids')

plt.title('Customer Segments (RFM)')
plt.xlabel('Recency (days since last purchase)')
plt.ylabel('Monetary (total spend)')
plt.legend(title='Segment', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rfm_segmentation_tableau_ready.png'), dpi=200)
plt.close()

# --- PCA Chart ---
plt.figure(figsize=(8,5))
for label, color in segment_palette.items():
    subset = rfm[rfm['SegmentLabel'] == label]
    plt.scatter(subset['PCA1'], subset['PCA2'], 
                label=label, color=color, alpha=0.8, edgecolor='k', s=60)

plt.title('Customer Segments (PCA Visualization)')
plt.xlabel('PCA1 (Customer Value Dimension)')
plt.ylabel('PCA2 (Engagement Dimension)')
plt.legend(title='Segment', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_segmentation_tableau_ready.png'), dpi=200)
plt.close()

# ============================================
# GPR: Recency + Frequency → Predict Monetary
# ============================================
print("🔥 Training Gaussian Process Regression (GPR)...")

X = rfm_scaled[:, :2]  # Recency & Frequency
y = rfm['Monetary'].values

kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0
