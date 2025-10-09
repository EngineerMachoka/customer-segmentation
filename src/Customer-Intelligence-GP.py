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
from mpl_toolkits.mplot3d import Axes3D
import joblib

# ============================================
# CONFIGURATION
# ============================================
AUTO_MODE = True  # ✅ Set True for automatic run (no manual input)

# === Paths ===
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
pca_plot_path = os.path.join(OUTPUT_DIR, 'pca_segmentation_v2.png')
csv_output_path = os.path.join(OUTPUT_DIR, 'customers_segmented_v2.csv')
xlsx_output_path = os.path.join(OUTPUT_DIR, 'customers_segmented_v2.xlsx')
metrics_output_path = os.path.join(OUTPUT_DIR, 'kmeans_metrics_v2.csv')
cluster_profile_path = os.path.join(OUTPUT_DIR, 'cluster_profiles_v2.csv')
scaler_path = os.path.join(OUTPUT_DIR, 'rfm_scaler.pkl')
model_path = os.path.join(OUTPUT_DIR, 'kmeans_model.pkl')
distribution_dir = os.path.join(OUTPUT_DIR, 'distributions')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(distribution_dir, exist_ok=True)

# ============================================
# DATA LOADING & CLEANING
# ============================================
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Data file not found: {file_path}")

print("📦 Loading data...")
data = pd.read_excel(file_path)
data.drop_duplicates(inplace=True)
data.dropna(subset=['CustomerID'], inplace=True)
data['total_spent'] = data['UnitPrice'] * data['Quantity']
data = data[data['total_spent'] > 0]
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
reference_date = data['InvoiceDate'].max()

# ============================================
# FEATURE ENGINEERING (RFM)
# ============================================
print("🧮 Creating RFM features...")
rfm = data.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('total_spent', 'sum')
).reset_index()

# Standardize RFM values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# ============================================
# FIND OPTIMAL CLUSTERS
# ============================================
print("🔍 Finding optimal K for KMeans...")
inertia, silhouette_scores = [], []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, labels))

metrics_df = pd.DataFrame({'k': list(k_range), 'Inertia': inertia, 'Silhouette': silhouette_scores})
metrics_df.to_csv(metrics_output_path, index=False)

# Auto-select best K
optimal_k = np.argmax(silhouette_scores) + 2
print(f"✅ Optimal number of clusters: {optimal_k}")

# ============================================
# FINAL KMEANS CLUSTERING
# ============================================
print("⚙️ Running final KMeans segmentation...")
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
joblib.dump(scaler, scaler_path)
joblib.dump(kmeans, model_path)

# ============================================
# PCA FOR VISUALIZATION
# ============================================
print("📊 Performing PCA for visualization...")
pca = PCA(n_components=2, random_state=42)
rfm_pca = pca.fit_transform(rfm_scaled)
rfm['PCA1'], rfm['PCA2'] = rfm_pca[:, 0], rfm_pca[:, 1]

# ============================================
# CLUSTER SUMMARY
# ============================================
cluster_summary = rfm.groupby('Segment').agg({
    'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean', 'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'}).round(1)

# Human-readable segment names
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
    plt.savefig(os.path.join(distribution_dir, f"{column}_distribution.png"))
    plt.close()

for col in ['Recency', 'Frequency', 'Monetary']:
    plot_distribution_with_iqr(rfm, col)

# ============================================
# GAUSSIAN PROCESS REGRESSION (GPR)
# ============================================
print("🔄 Training Gaussian Process Regression (GPR)...")
X = rfm_scaled[:, :2]  # Recency & Frequency
y = rfm['Monetary'].values

kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=3, random_state=42)
gpr.fit(X, y)
print(f"✅ GPR trained. Kernel: {gpr.kernel_}")

y_pred, y_std = gpr.predict(X, return_std=True)
rfm['GPR_Predicted_Monetary'] = y_pred
rfm['GPR_Uncertainty'] = y_std

# --- Visualization: Predicted Surface ---
r = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
f = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
R, F = np.meshgrid(r, f)
X_grid = np.column_stack([R.ravel(), F.ravel()])
Y_mean, Y_std = gpr.predict(X_grid, return_std=True)
Y_mean, Y_std = Y_mean.reshape(R.shape), Y_std.reshape(R.shape)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R, F, Y_mean, cmap='viridis', alpha=0.8)
ax.set_xlabel('Recency (scaled)')
ax.set_ylabel('Frequency (scaled)')
ax.set_zlabel('Predicted Monetary')
ax.set_title('GPR Predicted Monetary Surface')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'gpr_predicted_surface.png'))
plt.close()

# --- Visualization: Uncertainty Heatmap ---
plt.figure(figsize=(6,5))
plt.contourf(R, F, Y_std, levels=20, cmap='coolwarm')
plt.colorbar(label='Uncertainty (std)')
plt.xlabel('Recency (scaled)')
plt.ylabel('Frequency (scaled)')
plt.title('GPR Prediction Uncertainty')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'gpr_uncertainty_heatmap.png'))
plt.close()

print("📈 Saved GPR prediction and uncertainty visualizations.")

# ============================================
# EXPORT FOR TABLEAU
# ============================================
rfm.to_csv(csv_output_path, index=False)
rfm.to_excel(xlsx_output_path, index=False)

print(f"""
✅ RFM segmentation and GPR complete!
------------------------------------
📄 Tableau-ready files:
  - CSV:  {csv_output_path}
  - XLSX: {xlsx_output_path}

📊 Visualizations:
  - Distributions: {distribution_dir}
  - GPR Surface:   gpr_predicted_surface.png
  - Uncertainty:   gpr_uncertainty_heatmap.png

📁 Cluster Summary:
{cluster_summary[['Label', 'Recency', 'Frequency', 'Monetary', 'Count']]}
""")
