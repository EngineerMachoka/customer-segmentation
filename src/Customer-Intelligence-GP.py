# Code starts here — same as before until KMeans

# === Automatically select optimal k ===
optimal_k = np.argmax(silhouette_scores) + 2
print(f"✅ Automatically selected optimal_k = {optimal_k} based on silhouette score.")

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

# === PCA Variance Plot ===
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

# === Assign Human-Friendly Cluster Names ===
def assign_segment_labels(summary_df):
    labels = {}
    monetary_rank = summary_df['Monetary'].rank(method='min', ascending=False)
    recency_rank = summary_df['Recency'].rank(method='min')  # Lower is better
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

# === Plot RFM Clusters ===
sns.set(style='whitegrid')
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='SegmentLabel', palette='viridis', s=100, edgecolor='w')
plt.title('Customer Segments (RFM)')
plt.legend(title='Segment')
plt.tight_layout()
plt.savefig(rfm_plot_path)
plt.close()

# === Plot PCA Clusters ===
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=rfm,
    x='PCA1', y='PCA2',
    hue='SegmentLabel',
    palette='viridis',
    s=100,
    edgecolor='w'
)
plt.xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title('Customer Segments (PCA)')
plt.legend(title='Segment')
plt.tight_layout()
plt.savefig(pca_plot_path)
plt.close()

# === Bayesian Linear Regression: Monetary ~ Frequency ===
import pymc3 as pm
import arviz as az

print("🔎 Running Bayesian Linear Regression on Monetary ~ Frequency...")
x = rfm['Frequency'].values
y = rfm['Monetary'].values

with pm.Model() as linear_model:
    intercept = pm.Normal('Intercept', mu=0, sigma=10)
    slope = pm.Normal('Slope', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=10)

    mu = intercept + slope * x
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(2000, tune=1000, cores=1, target_accept=0.95, progressbar=False)

# Posterior prediction
x_pred = np.linspace(x.min(), x.max(), 100)
with linear_model:
    pm.set_data({'Frequency': x_pred})
    post_pred = pm.sample_posterior_predictive(trace, var_names=['Y_obs'], progressbar=False)

y_pred_mean = post_pred['Y_obs'].mean(axis=0)
y_pred_hpd = az.hdi(post_pred['Y_obs'], hdi_prob=0.95)

# Plot
plt.figure(figsize=(8,5))
plt.scatter(x, y, c='blue', alpha=0.5, label='Data')
plt.plot(x_pred, y_pred_mean, color='red', label='Mean regression line')
plt.fill_between(x_pred, y_pred_hpd[:,0], y_pred_hpd[:,1], color='red', alpha=0.3, label='95% Credible Interval')
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Bayesian Linear Regression: Monetary vs Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'bayesian_linear_regression.png'))
plt.close()

# === Export Clustered Data ===
rfm.to_csv(csv_output_path, index=False)

# === Final Summary ===
print("\n📊 Cluster Profiles:\n")
print(cluster_summary[['Label', 'Recency', 'Frequency', 'Monetary', 'Count']])
print(f"\n✅ Segmentation complete. Results saved to: {csv_output_path}")
