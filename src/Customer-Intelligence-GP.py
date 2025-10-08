import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import pymc3 as pm
import joblib

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
pca_variance_path = os.path.join(OUTPUT_DIR, 'pca_variance_v2.png')
rfm_plot_path = os.path.join(OUTPUT_DIR, 'rfm_segmentation_v2.png')
pca_plot_path = os.path.join(OUTPUT_DIR, 'pca_segmentation_v2.png')
gaussian_path = os.path.join(OUTPUT_DIR, 'gaussian_iqr_plot_v2.png')
bayesian_path = os.path.join(OUTPUT_DIR, 'bayesian_regression_v2.png')
csv_output_path = os.path.join(OUTPUT_DIR, 'customers_segmented_v2.csv')
metrics_output_path = os.path.join(OUTPUT_DIR, 'kmeans_metrics_v2.csv')
cluster_profile_path = os.path.join(OUTPUT_DIR, 'cluster_profiles_v2.csv')
scaler_path = os.path.join(OUTPUT_DIR, 'rfm_scaler.pkl')
model_path = os.path.join(OUTPUT_DIR, 'kmeans_model.pkl')

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

# === RFM Features ===
rfm = data.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('total_spent', 'sum')
).reset_index()

# === Standardization ===
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

# Save metrics
pd.DataFrame({'k': list(k_range), 'Inertia': inertia, 'Silhouette': silhouette_scores}).to_csv(metrics_output_path, index=False)

# Save Elbow & Silhouette plots
plt.figure(figsize=(6, 4))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.tight_layout()
plt.savefig(elbow_path)
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(k_range, silhouette_scores, marker='o', color='green')
plt.title('Silhouette Score')
plt.xlabel('k')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig(silhouette_path)
plt.close()

# === Auto-select optimal k ===
optimal_k = np.argmax(silhouette_scores) + 2
print(f"✅ Automatically selected optimal_k = {optimal_k} based on silhouette score.")

# === KMeans Clustering ===
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

joblib.dump(scaler, scaler_path)
joblib.dump(kmeans, model_path)

# === PCA ===
pca = PCA(n_components=2, random_state=42)
rfm_pca = pca.fit_transform(rfm_scaled)
rfm['PCA1'] = rfm_pca[:, 0]
rfm['PCA2'] = rfm_pca[:, 1]

plt.figure(figsize=(6, 4))
plt.bar(['PC1', 'PC2'], pca.explained_variance_ratio_, color='skyblue')
plt.title('PCA Variance Explained')
plt.tight_layout()
plt.savefig(pca_variance_path)
plt.close()

# === Cluster Summary & Labeling ===
summary = rfm.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'}).round(1)

def assign_labels(df):
    labels = {}
    for idx in df.index:
        if df.loc[idx, 'Monetary'] == df['Monetary'].max():
            labels[idx] = 'High Value'
        elif df.loc[idx, 'Frequency'] == df['Frequency'].max():
            labels[idx] = 'Loyal'
        elif df.loc[idx, 'Recency'] == df['Recency'].max():
            labels[idx] = 'Churn Risk'
        else:
            labels[idx] = 'Regular'
    return labels

labels = assign_labels(summary)
rfm['SegmentLabel'] = rfm['Segment'].map(labels)
summary['Label'] = summary.index.map(labels)
summary.to_csv(cluster_profile_path)

# === RFM Cluster Plot ===
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='SegmentLabel', palette='viridis', s=100, edgecolor='w')
plt.title('Customer Segments (RFM)')
plt.tight_layout()
plt.savefig(rfm_plot_path)
plt.close()

# === PCA Cluster Plot ===
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='SegmentLabel', palette='viridis', s=100, edgecolor='w')
plt.xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title('Customer Segments (PCA)')
plt.tight_layout()
plt.savefig(pca_plot_path)
plt.close()

# === Gaussian Distribution with IQR for Monetary ===
Q1 = rfm['Monetary'].quantile(0.25)
Q3 = rfm['Monetary'].quantile(0.75)
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5*IQR, Q3 + 1.5*IQR

plt.figure(figsize=(8, 4))
sns.histplot(rfm['Monetary'], kde=True, color='skyblue')
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower IQR Bound')
plt.axvline(upper_bound, color='red', linestyle='--', label='Upper IQR Bound')
plt.title('Monetary Distribution with IQR Outlier Bounds')
plt.legend()
plt.tight_layout()
plt.savefig(gaussian_path)
plt.close()

# === Bayesian Linear Regression (Monetary ~ Frequency) ===
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    mu = alpha + beta * rfm['Frequency']
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=rfm['Monetary'])

    trace = pm.sample(1000, tune=1000, cores=1, progressbar=False)

# Plot posterior predictive regression line
freq_range = np.linspace(rfm['Frequency'].min(), rfm['Frequency'].max(), 100)
pred_monetary = trace['alpha'].mean() + trace['beta'].mean() * freq_range

plt.figure(figsize=(8, 5))
plt.scatter(rfm['Frequency'], rfm['Monetary'], alpha=0.5, label='Data')
plt.plot(freq_range, pred_monetary, color='red', label='Bayesian Regression')
plt.title('Bayesian Linear Regression: Monetary ~ Frequency')
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.legend()
plt.tight_layout()
plt.savefig(bayesian_path)
plt.close()

# === Export Final Data ===
rfm.to_csv(csv_output_path, index
