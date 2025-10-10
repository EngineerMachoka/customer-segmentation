import os  # File operations
import random  # Random numbers for renaming
import pandas as pd  # Data manipulation and handling
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Advanced visualization
from sklearn.preprocessing import StandardScaler  # Feature scaling for clustering
from sklearn.cluster import KMeans  # KMeans clustering algorithm
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score  # Evaluation metrics
from sklearn.decomposition import PCA  # Dimensionality reduction
from sklearn.gaussian_process import GaussianProcessRegressor  # Gaussian Process Regression
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel  # GPR kernels
from scipy.stats import norm  # Gaussian fitting
import joblib  # Save/load models
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

# Configuration paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current script folder
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))  # Project root folder
DATA_DIR = os.path.join(BASE_DIR, 'data')  # Input data folder
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')  # Output folder
DISTRIBUTION_DIR = os.path.join(OUTPUT_DIR, 'distributions')  # Distribution plots folder

os.makedirs(DATA_DIR, exist_ok=True)  # Ensure input folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output folder exists
os.makedirs(DISTRIBUTION_DIR, exist_ok=True)  # Ensure distribution folder exists

# Rename existing files to avoid overwrite
def rename_existing_files(folder):  # Function to rename files with random prefix
    for root, dirs, files in os.walk(folder):  # Iterate through all files in folder
        for filename in files:
            old_path = os.path.join(root, filename)  # Original file path
            rand_prefix = str(random.randint(1000, 9999))  # Generate random prefix
            new_filename = f"{rand_prefix}_{filename}"  # Create new filename
            new_path = os.path.join(root, new_filename)  # New file path
            try:
                os.rename(old_path, new_path)  # Rename file
                print(f"📂 Renamed existing file → {new_filename}")  # Print confirmation
            except Exception as e:
                print(f"⚠️ Could not rename {filename}: {e}")  # Handle errors

rename_existing_files(OUTPUT_DIR)  # Execute renaming to avoid overwriting outputs

# Load data
DATA_FILE = 'Online Retail.xlsx'  # Data filename
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)  # Full data path
if not os.path.exists(DATA_PATH):  # Check if file exists
    raise FileNotFoundError(f"❌ Data file not found: {DATA_PATH}")  # Raise error if missing

print("📦 Loading dataset...")  # Notify loading
data = pd.read_excel(DATA_PATH)  # Load Excel data
data.drop_duplicates(inplace=True)  # Remove duplicates
data.dropna(subset=['CustomerID'], inplace=True)  # Remove missing CustomerID
data['total_spent'] = data['UnitPrice'] * data['Quantity']  # Calculate total spending
data = data[data['total_spent'] > 0]  # Remove non-positive spend
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])  # Convert to datetime
reference_date = data['InvoiceDate'].max()  # Latest date as reference

# Compute RFM metrics
print("🧮 Computing RFM metrics...")  # Notify RFM computation
rfm = data.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),  # Days since last purchase
    Frequency=('InvoiceNo', 'nunique'),  # Number of unique invoices
    Monetary=('total_spent', 'sum')  # Total spending per customer
).reset_index()  # Reset index to DataFrame

scaler = StandardScaler()  # Initialize scaler
rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])  # Standardize RFM

# Determine optimal clusters
print("🔍 Determining optimal cluster count...")  # Notify cluster selection
inertia = []  # List for inertia values
silhouette_scores = []  # List for silhouette scores
for k in range(2,11):  # Test clusters from 2 to 10
    km = KMeans(n_clusters=k, n_init=10, random_state=42)  # Initialize KMeans
    labels = km.fit_predict(rfm_scaled)  # Fit and predict clusters
    inertia.append(km.inertia_)  # Append inertia
    silhouette_scores.append(silhouette_score(rfm_scaled, labels))  # Append silhouette score
metrics_df = pd.DataFrame({'k':range(2,11),'Inertia':inertia,'Silhouette':silhouette_scores})  # Save metrics
metrics_df.to_csv(os.path.join(OUTPUT_DIR,'kmeans_metrics_full.csv'), index=False)  # Export metrics
optimal_k = np.argmax(silhouette_scores)+2  # Best cluster based on silhouette
print(f"✅ Optimal clusters determined: {optimal_k}")  # Print result

# Final KMeans clustering
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)  # Initialize final KMeans
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)  # Assign cluster labels
centroids = scaler.inverse_transform(kmeans.cluster_centers_)  # Centroids in original scale
joblib.dump(scaler, os.path.join(OUTPUT_DIR,'rfm_scaler.pkl'))  # Save scaler
joblib.dump(kmeans, os.path.join(OUTPUT_DIR,'kmeans_model.pkl'))  # Save model

# PCA visualization
pca = PCA(n_components=2, random_state=42)  # Initialize PCA
rfm_pca = pca.fit_transform(rfm_scaled)  # Fit PCA
rfm['PCA1'], rfm['PCA2'] = rfm_pca[:,0], rfm_pca[:,1]  # Store PCA components

# Cluster summary with tuple-based aggregation
cluster_summary = rfm.groupby('Segment').agg(
    Recency=('Recency','mean'),  # Average recency per cluster
    Frequency=('Frequency','mean'),  # Average frequency
    Monetary=('Monetary','mean'),  # Average monetary
    Count=('CustomerID','count')  # Count customers per cluster
).round(1)  # Round to 1 decimal

# Assign human-readable labels
def assign_segment_labels(summary_df):  # Function to label clusters
    labels = {}  # Dictionary for labels
    monetary_rank = summary_df['Monetary'].rank(method='min', ascending=False)  # Rank monetary
    recency_rank = summary_df['Recency'].rank(method='min')  # Rank recency
    frequency_rank = summary_df['Frequency'].rank(method='min', ascending=False)  # Rank frequency
    for idx in summary_df.index:  # Iterate clusters
        if monetary_rank[idx]==1:
            labels[idx]='High Value'  # Top monetary
        elif frequency_rank[idx]==1:
            labels[idx]='Loyal'  # Top frequency
        elif recency_rank[idx]==summary_df['Recency'].rank().max():
            labels[idx]='Churn Risk'  # Highest recency
        else:
            labels[idx]='Regular'  # Default label
    return labels  # Return mapping

segment_labels = assign_segment_labels(cluster_summary)  # Generate labels
rfm['SegmentLabel'] = rfm['Segment'].map(segment_labels)  # Map to RFM
cluster_summary['Label'] = cluster_summary.index.map(segment_labels)  # Map to summary
cluster_summary.to_csv(os.path.join(OUTPUT_DIR,'cluster_profiles_full.csv'), index=False)  # Export

# Distribution plot functions
def plot_distribution_with_iqr(data,column):  # Function for IQR distribution plot
    plt.figure(figsize=(7,4))  # Figure size
    sns.histplot(data[column], bins=30, kde=True, color='skyblue', edgecolor='black', stat='density')  # Histogram
    mu,std = norm.fit(data[column])  # Fit Gaussian
    x=np.linspace(data[column].min(),data[column].max(),100)  # X values for Gaussian
    plt.plot(x,norm.pdf(x,mu,std),'r--',label='Gaussian Fit')  # Gaussian overlay
    q1,q3 = np.percentile(data[column],[25,75])  # Compute IQR
    plt.axvline(q1,color='green',linestyle='--',label='Q1')  # Q1 line
    plt.axvline(q3,color='purple',linestyle='--',label='Q3')  # Q3 line
    plt.title(f'{column} Distribution')  # Title
    plt.legend()  # Show legend
    plt.tight_layout()  # Adjust layout
    plt.savefig(os.path.join(DISTRIBUTION_DIR,f"{column}_distribution.png"), dpi=200)  # Save plot
    plt.close()  # Close figure

def plot_all_distributions(data,column):  # Generate full distribution set
    q1,q3 = np.percentile(data[column],[25,75])  # Compute quartiles
    full = data[column]  # Full data
    iqr = full[(full>=q1)&(full<=q3)]  # Middle 50%
    extremes = pd.concat([full[full<=q1], full[full>=q3]])  # Extremes
    top25 = full[full>=q3]  # Top quartile
    bottom25 = full[full<=q1]  # Bottom quartile

    # Full vs IQR
    fig,axes = plt.subplots(1,2,figsize=(10,4))  # Two panels
    sns.histplot(full,bins=30,kde=True,color='skyblue',ax=axes[0],edgecolor='black',stat='density')  # Full
    axes[0].set_title('Full')  # Title
    sns.histplot(iqr,bins=20,kde=True,color='orange',ax=axes[1],edgecolor='black',stat='density')  # IQR
    axes[1].set_title('IQR (Middle 50%)')  # Title
    plt.tight_layout()  # Adjust layout
    plt.savefig(os.path.join(DISTRIBUTION_DIR,f"{column}_iqr_comparison.png"),dpi=200)  # Save
    plt.close()  # Close

    # Extremes vs Full
    fig,axes = plt.subplots(1,3,figsize=(14,4))  # Three panels
    sns.histplot(full,bins=30,kde=True,color='skyblue',ax=axes[0],edgecolor='black',stat='density')  # Full
    axes[0].set_title('Full')  # Title
    sns.histplot(iqr,bins=20,kde=True,color='orange',ax=axes[1],edgecolor='black',stat='density')  # IQR
    axes[1].set_title('IQR')  # Title
    sns.histplot(extremes,bins=20,kde=True,color='crimson',ax=axes[2],edgecolor='black',stat='density')  # Extremes
    axes[2].set_title('Extremes (Top+Bottom 25%)')  # Title
    plt.tight_layout()  # Adjust layout
    plt.savefig(os.path.join(DISTRIBUTION_DIR,f"{column}_extremes_contrast.png"),dpi=200)  # Save
    plt.close()  # Close

    # Top25
    plt.figure(figsize=(6,4))  # Figure
    sns.histplot(top25,bins=20,kde=True,color='red',edgecolor='black',stat='density')  # Top25 histogram
    plt.title('Top25 High Spenders')  # Title
    plt.tight_layout()  # Layout
    plt.savefig(os.path.join(DISTRIBUTION_DIR,f"{column}_top25.png"),dpi=200)  # Save
    plt.close()  # Close

    # Bottom25
    plt.figure(figsize=(6,4))  # Figure
    sns.histplot(bottom25,bins=20,kde=True,color='blue',edgecolor='black',stat='density')  # Bottom25 histogram
    plt.title('Bottom25 Low Value')  # Title
    plt.tight_layout()  # Layout
    plt.savefig(os.path.join(DISTRIBUTION_DIR,f"{column}_bottom25.png"),dpi=200)  # Save
    plt.close()  # Close

# Generate plots
for col in ['Recency','Frequency','Monetary']:  # Iterate RFM columns
    plot_distribution_with_iqr(rfm,col)  # Basic IQR distribution
    plot_all_distributions(rfm,col)  # Full set of plots
