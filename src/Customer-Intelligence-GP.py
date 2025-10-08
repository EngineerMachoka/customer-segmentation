import os  # For handling file paths and directories
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For enhanced statistical plotting
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.cluster import KMeans  # For clustering analysis
from sklearn.metrics import silhouette_score  # For evaluating clustering quality
from sklearn.decomposition import PCA  # For dimensionality reduction
from sklearn.gaussian_process import GaussianProcessRegressor  # For Gaussian Process regression
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel  # Kernels for GP
import joblib  # For saving/loading models and scalers

# === Define directories ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current script directory
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))  # Base directory (one level up)
DATA_DIR = os.path.join(BASE_DIR, 'data')  # Data folder path
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')  # Outputs folder path

# === File names ===
DATA_FILE = 'Online Retail.xlsx'  # Data file name
file_path = os.path.join(DATA_DIR, DATA_FILE)  # Full path to data file

# Paths to output files (plots, CSVs, models)
elbow_path = os.path.join(OUTPUT_DIR, 'elbow_plot_v2.png')
silhouette_path = os.path.join(OUTPUT_DIR, 'silhouette_plot_v2.png')
pca_variance_path = os.path.join(OUTPUT_DIR, 'pca_variance_v2.png')
rfm_plot_path = os.path.join(OUTPUT_DIR, 'rfm_segmentation_v2.png')
pca_plot_path = os.path.join(OUTPUT_DIR, 'pca_segmentation_v2.png')
gaussian_iqr_path = os.path.join(OUTPUT_DIR, 'gaussian_iqr_plot_v2.png')
gaussian_process_path = os.path.join(OUTPUT_DIR, 'gaussian_process_regression.png')
csv_output_path = os.path.join(OUTPUT_DIR, 'customers_segmented_v2.csv')
metrics_output_path = os.path.join(OUTPUT_DIR, 'kmeans_metrics_v2.csv')
cluster_profile_path = os.path.join(OUTPUT_DIR, 'cluster_profiles_v2.csv')
scaler_path = os.path.join(OUTPUT_DIR, 'rfm_scaler.pkl')
model_path = os.path.join(OUTPUT_DIR, 'kmeans_model.pkl')

os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist

# === Load and clean data ===
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")  # Raise error if data file missing

data = pd.read_excel(file_path)  # Read Excel data into DataFrame
data.drop_duplicates(inplace=True)  # Remove duplicate rows
data.dropna(subset=['CustomerID'], inplace=True)  # Drop rows without CustomerID
data['total_spent'] = data['UnitPrice'] * data['Quantity']  # Calculate total spent per row
data = data[data['total_spent'] > 0]  # Keep only positive total spent values
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])  # Convert InvoiceDate to datetime
reference_date = data['InvoiceDate'].max()  # Use latest date as reference for recency

# === Calculate RFM features per customer ===
rfm = data.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),  # Days since last purchase
    Frequency=('InvoiceNo', 'nunique'),  # Number of unique invoices
    Monetary=('total_spent', 'sum')  # Total amount spent
).reset_index()

# === Scale RFM features for clustering ===
scaler = StandardScaler()  # Initialize standard scaler
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])  # Scale RFM data

# === Find optimal number of clusters using Elbow and Silhouette methods ===
inertia = []  # Sum of squared distances to cluster centers (for elbow plot)
silhouette_scores = []  # Silhouette scores for cluster quality
k_range = range(2, 11)  # Range of k to try (2 to 10 clusters)

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)  # Initialize KMeans
    labels = km.fit_predict(rfm_scaled)  # Fit and predict cluster labels
    inertia.append(km.inertia_)  # Store inertia
    silhouette_scores.append(silhouette_score(rfm_scaled, labels))  # Store silhouette score

# Save clustering metrics to CSV
pd.DataFrame({'k': list(k_range), 'Inertia': inertia, 'Silhouette': silhouette_scores}).to_csv(metrics_output_path, index=False)

# Plot Elbow curve (Inertia vs k)
plt.figure(figsize=(6, 4))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia')
plt.tight_layout()
plt.savefig(elbow_path)  # Save plot
plt.close()

# Plot Silhouette scores vs k
plt.figure(figsize=(6, 4))
plt.plot(k_range, silhouette_scores, marker='o', color='green')
plt.title('Silhouette Score')
plt.xlabel('Number of clusters k')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig(silhouette_path)  # Save plot
plt.close()

# === Automatically select k with highest silhouette score ===
optimal_k = np.argmax(silhouette_scores) + 2  # +2 because k starts at 2
print(f"Automatically selected optimal_k = {optimal_k} based on silhouette score.")

# === Fit final KMeans clustering model with optimal_k ===
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)  # Assign cluster labels
centroids = scaler.inverse_transform(kmeans.cluster_centers_)  # Convert centroids back to original scale

# Save scaler and model to disk for future use
joblib.dump(scaler, scaler_path)
joblib.dump(kmeans, model_path)

# === Apply PCA for 2D visualization ===
pca = PCA(n_components=2, random_state=42)
rfm_pca = pca.fit_transform(rfm_scaled)  # Transform scaled data to 2 principal components
rfm['PCA1'] = rfm_pca[:, 0]  # Add first principal component as new column
rfm['PCA2'] = rfm_pca[:, 1]  # Add second principal component as new column

# Plot explained variance ratio by the two PCs
plt.figure(figsize=(6, 4))
plt.bar(['PC1', 'PC2'], pca.explained_variance_ratio_, color='skyblue')
plt.title('PCA Variance Explained')
plt.tight_layout()
plt.savefig(pca_variance_path)
plt.close()

# === Create summary statistics per cluster ===
summary = rfm.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'}).round(1)  # Round for readability

# === Assign human-readable labels for segments based on summary stats ===
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

labels = assign_labels(summary)  # Get labels per cluster index
rfm['SegmentLabel'] = rfm['Segment'].map(labels)  # Map numeric cluster to label
summary['Label'] = summary.index.map(labels)  # Add labels to summary
summary.to_csv(cluster_profile_path)  # Save cluster profiles summary

# === Plot RFM segments on Recency vs Monetary space ===
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='SegmentLabel', palette='viridis', s=100, edgecolor='w')
plt.title('Customer Segments (RFM)')
plt.tight_layout()
plt.savefig(rfm_plot_path)
plt.close()

# === Plot PCA segments on first two principal components ===
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='SegmentLabel', palette='viridis', s=100, edgecolor='w')
plt.xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title('Customer Segments (PCA)')
plt.tight_layout()
plt.savefig(pca_plot_path)
plt.close()

# === Plot Monetary distribution with IQR outlier bounds using multiplier 0.74 ===
def plot_iqr_gaussian(df, col, multiplier, output_path):
    Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - multiplier * IQR  # Lower fence for outliers
    upper_bound = Q3 + multiplier * IQR  # Upper fence for outliers
    
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, color='skyblue')  # Histogram with KDE
    plt.axvline(lower_bound, color='red', linestyle='--', label='Lower IQR Bound')
    plt.axvline(upper_bound, color='red', linestyle='--', label='Upper IQR Bound')
    plt.title(f'{col} Distribution with IQR Outlier Bounds (multiplier={multiplier})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Call function to plot Monetary distribution with custom multiplier
plot_iqr_gaussian(rfm, 'Monetary', 0.74, gaussian_iqr_path)

# === Gaussian Process Regression for Monetary ~ Frequency with uncertainty ===
def gaussian_process_regression(X, y, output_path):
    # Define kernel: Constant * RBF (smoothness) + WhiteKernel (noise)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)  # Initialize GP regressor
    gp.fit(X.reshape(-1, 1), y)  # Fit GP on data
    
    # Prepare grid for prediction over range of X values
    X_pred = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_pred, sigma = gp.predict(X_pred, return_std=True)  # Predict mean and std deviation
    
    # Plot original data, mean prediction and confidence interval
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, alpha=0.5, label='Data')
    plt.plot(X_pred, y_pred, 'r-', label='GP Mean Prediction')
    plt.fill_between(X_pred.flatten(), y_pred - 1.96*sigma, y_pred + 1.96*sigma,
                     alpha=0.3, color='r', label='95% Confidence Interval')
    plt.title('Gaussian Process Regression with Uncertainty')
    plt.xlabel('Frequency')
    plt.ylabel('Monetary')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Run Gaussian Process regression and plot results
gaussian_process_regression(rfm['Frequency'].values, rfm['Monetary'].values, gaussian_process_path)

# === Export final RFM data with clusters and labels to CSV ===
rfm.to_csv(csv_output_path, index=False)
