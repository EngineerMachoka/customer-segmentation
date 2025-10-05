import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# === Define paths relative to this script ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# === Filenames ===
DATA_FILE = 'Online Retail.xlsx'
file_path = os.path.join(DATA_DIR, DATA_FILE)

# Output files with _v2 to avoid overwrite
elbow_path = os.path.join(OUTPUT_DIR, 'elbow_plot_v2.png')
silhouette_path = os.path.join(OUTPUT_DIR, 'silhouette_plot_v2.png')
pca_variance_path = os.path.join(OUTPUT_DIR, 'pca_variance_v2.png')
rfm_plot_path = os.path.join(OUTPUT_DIR, 'rfm_segmentation_v2.png')
pca_plot_path = os.path.join(OUTPUT_DIR, 'pca_segmentation_v2.png')
csv_output_path = os.path.join(OUTPUT_DIR, 'customers_segmented_v2.csv')

# === Ensure output folder exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Data ===
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

print("Reading data...")
data = pd.read_excel(file_path)
data.drop_duplicates(inplace=True)
data.dropna(subset=['CustomerID'], inplace=True)

data['total_spent'] = data['UnitPrice'] * data['Quantity']
data = data[data['total_spent'] > 0]

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
reference_date = data['InvoiceDate'].max()

# === Create RFM Features ===
print("🔧 Building RFM features...")
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
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, labels))

# Elbow Plot
plt.figure(figsize=(6, 4))
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.tight_layout()
plt.savefig(elbow_path)
plt.close()

# Silhouette Plot
plt.figure(figsize=(6, 4))
plt.plot(range(2, 11), silhouette_scores, marker='o', color='green')
plt.title('Silhouette Score')
plt.xlabel('k')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig(silhouette_path)
plt.close()

# === Final KMeans Clustering ===
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# === PCA for Visualization ===
pca = PCA(n_components=2, random_state=42)
rfm_pca = pca.fit_transform(rfm_scaled)
rfm['PCA1'] = rfm_pca[:, 0]
rfm['PCA2'] = rfm_pca[:, 1]

# === PCA Variance Plot ===
plt.figure(figsize=(6, 4))
plt.bar(['PC1', 'PC2'], pca.explained_variance_ratio_, color='skyblue')
plt.title('PCA Variance Explained')
plt.tight_layout()
plt.savefig(pca_variance_path)
plt.close()

# === Plot RFM Clusters ===
sns.set(style='whitegrid')
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment', palette='viridis', s=100, edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 2], c='red', s=300, marker='X', label='Centroids')
plt.title('Customer Segments (RFM)')
plt.legend()
plt.tight_layout()
plt.savefig(rfm_plot_path)
plt.close()

# === Plot PCA Clusters ===
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='Segment', palette='viridis', s=100, edgecolor='k')
plt.title('Customer Segments (PCA)')
plt.legend()
plt.tight_layout()
plt.savefig(pca_plot_path)
plt.close()

# === Export CSV ===
rfm.to_csv(csv_output_path, index=False)
print(f"Segmentation complete. Results saved to:\n{csv_output_path}")
