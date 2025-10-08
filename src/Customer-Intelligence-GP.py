import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib
import pymc3 as pm
import arviz as az

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
bayesian_plot_path = os.path.join(OUTPUT_DIR, 'bayesian_linear_regression.png')

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Data ===
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

print("📥 Reading data...")
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
k_range = range(2, 11)
print("📊 Calculating Elbow and Silhouette scores...")
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, labels))

# Save metrics
metrics_df = pd.DataFrame({
    'k': list(k_range),
    'Inertia': inertia,
    'Silhouette': silhouette_scores
})
metrics_df.to_csv(metrics_output_path, index=False)

# Plots
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

# === Automatically determine optimal k ===
optimal_k = np.argmax(silhouette_scores) + 2
print(f"✅ Automatically selected optimal_k = {optimal_k} based on silhouette score.")

# === Final Clustering ===
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

# PCA Variance Plot
plt.figure(figsize=(6, 4))
plt.bar(['PC1', 'PC2'], pca.explained_variance_ratio_, color='skyblue')
plt.title('PCA Variance Explained')
plt.tight_layout()
plt.savefig(pca_variance_path)
plt.close()

# === Cluster Summary ===
cluster_summary = rfm.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'}).round(1)

# === Assign Cluster Labels ===
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

# === RFM Cluster Plot ===
sns.set(style='whitegrid')
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='SegmentLabel', palette='viridis', s=100, edgecolor='w')
plt.title('Customer Segments (RFM)')
plt.legend(title='Segment')
plt.tight_layout()
plt.savefig(rfm_plot_path)
plt.close()

# === PCA Cluster Plot ===
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='SegmentLabel', palette='viridis', s=100, edgecolor='w')
plt.xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title('Customer Segments (PCA)')
plt.legend(title='Segment')
plt.tight_layout()
plt.savefig(pca_plot_path)
plt.close()

# === Bayesian Linear Regression (Monetary ~ Frequency) ===
print("📈 Running Bayesian Linear Regression...")
x = rfm['Frequency'].values
y = rfm['Monetary'].values

with pm.Model() as model:
    intercept = pm.Normal('Intercept', mu=0, sigma=10)
    slope = pm.Normal('Slope', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=10)

    mu = intercept + slope * x
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(2000, tune=1000, target_accept=0.95, cores=1, progressbar=False)

# Prediction
x_pred = np.linspace(x.min(), x.max(), 100)
with model:
    post_pred = pm.sample_posterior_predictive(trace, var_names=['Y_obs'], keep_size=True, progressbar=False)

y_pred_mean = post_pred['Y_obs'].mean(axis=0)
y_pred_hpd = az.hdi(post_pred['Y_obs'], hdi_prob=0.95)

# Plot Bayesian Regression
plt.figure(figsize=(8, 5))
plt.scatter(x, y, c='blue', alpha=0.5, label='Observed')
plt.plot(x_pred, y_pred_mean[:100], color='red', label='Mean Prediction')
plt.fill_between(x_pred, y_pred_hpd[:100, 0], y_pred_hpd[:100, 1], color='red', alpha=0.3, label='95% CI')
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Bayesian Linear Regression: Monetary ~ Frequency')
plt.legend()
plt.tight_layout()
plt
