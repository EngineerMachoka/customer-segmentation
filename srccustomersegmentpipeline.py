import os
import logging
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# === Configuration (can be replaced by YAML/JSON in real projects) ===
class Config:
    input_file = 'Online Retail.xlsx'
    output_dir = 'outputs'
    optimal_k = 4
    random_state = 42

# === Utility functions ===
def ensure_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

# === Load and preprocess data ===
def load_data(filepath: str) -> pd.DataFrame:
    logging.info(f"Loading data from {filepath} ...")
    df = pd.read_excel(filepath)
    logging.info(f"Initial data shape: {df.shape}")

    df.drop_duplicates(inplace=True)
    df.dropna(subset=['CustomerID'], inplace=True)

    df['total_spent'] = df['UnitPrice'] * df['Quantity']
    df = df[df['total_spent'] > 0]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    logging.info(f"Data cleaned: {df.shape}")
    return df

# === Create RFM features ===
def create_rfm(df: pd.DataFrame) -> pd.DataFrame:
    reference_date = df['InvoiceDate'].max()
    logging.info(f"Reference date for recency: {reference_date}")

    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('total_spent', 'sum')
    ).reset_index()
    logging.info(f"RFM features created: {rfm.shape}")
    return rfm

# === Scale features ===
def scale_features(df: pd.DataFrame, features: list) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    logging.info(f"Features scaled: {features}")
    return scaled, scaler

# === Calculate elbow and silhouette scores ===
def evaluate_clusters(data_scaled: np.ndarray, k_min=2, k_max=10) -> Tuple[list, list]:
    inertia = []
    silhouette_scores = []
    logging.info("Calculating inertia and silhouette scores...")
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=Config.random_state)
        labels = kmeans.fit_predict(data_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data_scaled, labels))
        logging.info(f"k={k}: inertia={kmeans.inertia_:.2f}, silhouette={silhouette_scores[-1]:.4f}")
    return inertia, silhouette_scores

# === Plot and save figures ===
def plot_and_save_elbow(inertia: list, path: str):
    plt.figure()
    plt.plot(range(2, 2 + len(inertia)), inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logging.info(f"Elbow plot saved: {path}")

def plot_and_save_silhouette(scores: list, path: str):
    plt.figure()
    plt.plot(range(2, 2 + len(scores)), scores, marker='o', color='green')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logging.info(f"Silhouette plot saved: {path}")

def plot_and_save_pca_variance(pca: PCA, path: str):
    plt.figure()
    plt.bar(['PC1', 'PC2'], pca.explained_variance_ratio_, color='skyblue')
    plt.title('PCA Variance Explained')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logging.info(f"PCA variance plot saved: {path}")

def plot_and_save_clusters(rfm: pd.DataFrame, centroids: np.ndarray, path_rfm: str, path_pca: str):
    sns.set(style='whitegrid')

    # RFM plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=rfm,
        x='Recency', y='Monetary',
        hue='Segment',
        palette='viridis',
        s=100,
        edgecolor='k'
    )
    plt.scatter(centroids[:, 0], centroids[:, 2], c='red', s=300, marker='X', label='Centroids')
    plt.title('Customer Segments (RFM)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_rfm)
    plt.close()
    logging.info(f"RFM cluster plot saved: {path_rfm}")

    # PCA plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=rfm,
        x='PCA1', y='PCA2',
        hue='Segment',
        palette='viridis',
        s=100,
        edgecolor='k'
    )
    plt.title('Customer Segments (PCA)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_pca)
    plt.close()
    logging.info(f"PCA cluster plot saved: {path_pca}")

# === Main pipeline function ===
def run_pipeline(config: Config):
    ensure_dir(config.output_dir)

    # Load and preprocess
    df = load_data(config.input_file)

    # RFM features
    rfm = create_rfm(df)

    # Scale
    rfm_scaled, scaler = scale_features(rfm, ['Recency', 'Frequency', 'Monetary'])

    # Evaluate clusters
    inertia, silhouette_scores = evaluate_clusters(rfm_scaled)

    # Save evaluation plots
    plot_and_save_elbow(inertia, os.path.join(config.output_dir, 'elbow_plot_v2.png'))
    plot_and_save_silhouette(silhouette_scores, os.path.join(config.output_dir, 'silhouette_plot_v2.png'))

    # Clustering with optimal k
    kmeans = KMeans(n_clusters=config.optimal_k, n_init=10, random_state=config.random_state)
    rfm['Segment'] = kmeans.fit_predict(rfm_scaled)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=config.random_state)
    rfm_pca = pca.fit_transform(rfm_scaled)
    rfm['PCA1'] = rfm_pca[:, 0]
    rfm['PCA2'] = rfm_pca[:, 1]

    # Save PCA variance plot
    plot_and_save_pca_variance(pca, os.path.join(config.output_dir, 'pca_variance_v2.png'))

    # Save cluster plots
    plot_and_save_clusters(
        rfm,
        centroids,
        os.path.join(config.output_dir, 'rfm_segmentation_v2.png'),
        os.path.join(config.output_dir, 'pca_segmentation_v2.png')
    )

    # Export results
    output_csv = os.path.join(config.output_dir, 'customers_segmented_v2.csv')
    rfm.to_csv(output_csv, index=False)
    logging.info(f"Segmentation complete. Results saved to: {output_csv}")

    return rfm, output_csv

# === Run the pipeline ===
if __name__ == '__main__':
    # For Colab, make sure to upload the file before running this cell
    try:
        results, csv_path = run_pipeline(Config)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
