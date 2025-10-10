"""
Customer Segmentation & Predictive Modeling – Fully Commented v9
---------------------------------------------------------------
This script:
• Computes RFM metrics
• Performs KMeans clustering
• Reduces dimensions using PCA
• Generates distribution plots (Full, IQR, Extremes, Top25, Bottom25)
• Trains Gaussian Process Regression (GPR)
• Generates automated GPR surface plots highlighting Top25/Bottom25
• Renames existing output files to prevent overwriting
• Contains inline + block comments explaining every step
"""

# ============================
# IMPORT LIBRARIES
# ============================
import os  # For file/directory operations
import random  # For generating random numbers for file renaming
import pandas as pd  # For data manipulation
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # Enhanced plotting features
from sklearn.preprocessing import StandardScaler  # Standardize features
from sklearn.cluster import KMeans  # Clustering algorithm
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score  # Model evaluation metrics
from sklearn.decomposition import PCA  # Dimensionality reduction
from sklearn.gaussian_process import GaussianProcessRegressor  # Gaussian Process Regression
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel  # GPR kernels
from scipy.stats import norm  # For Gaussian fitting
import joblib  # For saving/loading models
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

# ============================
# CONFIGURATION & PATHS
# ============================
AUTO_MODE = True  # If True, script runs automatically without manual input

# Define directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current script folder
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))  # Project root folder
DATA_DIR = os.path.join(BASE_DIR, 'data')  # Input data folder
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')  # Output folder
DISTRIBUTION_DIR = os.path.join(OUTPUT_DIR, 'distributions')  # Folder for distribution plots

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DISTRIBUTION_DIR, exist_ok=True)

# ============================
# RENAME EXISTING FILES TO AVOID OVERWRITE
# ============================
def rename_existing_files(folder):
    """
    Rename all existing files in a folder by adding a random numeric prefix.
    This prevents overwriting previously generated outputs.
    """
    for root, dirs, files in os.walk(folder):  # Traverse all files in folder
        for filename in files:  # Loop through each file
            old_path = os.path.join(root, filename)  # Original file path
            rand_prefix = str(random.randint(1000, 9999))  # Random 4-digit prefix
            new_filename = f"{rand_prefix}_{filename}"  # New file name
            new_path = os.path.join(root, new_filename)  # New full path
            try:
                os.rename(old_path, new_path)  # Rename file
                print(f"📂 Renamed existing file → {new_filename}")  # Confirmation
            except Exception as e:  # Catch errors
                print(f"⚠️ Could not rename {filename}: {e}")  # Error message

# Apply renaming to OUTPUT_DIR
rename_existing_files(OUTPUT_DIR)

# ============================
# LOAD & CLEAN DATA
# ============================
DATA_FILE = 'Online Retail.xlsx'  # Input data filename
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)  # Full path to input file

# Check if the file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Data file not found: {DATA_PATH}")  # Stop execution if missing

print("📦 Loading dataset...")  # Inform user

# Load Excel data into pandas DataFrame
data = pd.read_excel(DATA_PATH)
data.drop_duplicates(inplace=True)  # Remove duplicate rows
data.dropna(subset=['CustomerID'], inplace=True)  # Remove rows missing CustomerID

# Calculate total spend per transaction
data['total_spent'] = data['UnitPrice'] * data['Quantity']

# Keep only positive spend transactions
data = data[data['total_spent'] > 0]

# Convert InvoiceDate to datetime format
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Reference date is the latest invoice date
reference_date = data['InvoiceDate'].max()

# ============================
# COMPUTE RFM METRICS
# ============================
print("🧮 Computing RFM metrics...")

# Aggregate data by CustomerID
rfm = data.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),  # Days since last purchase
    Frequency=('InvoiceNo', 'nunique'),  # Count of unique invoices
    Monetary=('total_spent', 'sum')  # Total monetary spend
).reset_index()  # Reset index to make CustomerID a column

# Standardize RFM features
scaler = StandardScaler()  # Initialize scaler
rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])  # Scale features

# ============================
# FIND OPTIMAL NUMBER OF CLUSTERS
# ============================
print("🔍 Determining optimal cluster count...")

# Initialize lists to store KMeans metrics
inertia = []
silhouette_scores = []

# Loop through cluster counts from 2 to 10
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)  # KMeans instance
    labels = km.fit_predict(rfm_scaled)  # Fit model and assign cluster labels
    inertia.append(km.inertia_)  # Record inertia
    silhouette_scores.append(silhouette_score(rfm_scaled, labels))  # Record silhouette score

# Save KMeans metrics to CSV
metrics_df = pd.DataFrame({'k':range(2,11),'Inertia':inertia,'Silhouette':silhouette_scores})
metrics_df.to_csv(os.path.join(OUTPUT_DIR,'kmeans_metrics_v9.csv'), index=False)

# Select optimal K based on highest silhouette score
optimal_k = np.argmax(silhouette_scores) + 2
print(f"✅ Optimal clusters determined: {optimal_k}")

# ============================
# FINAL KMEANS CLUSTERING
# ============================
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)  # Initialize KMeans
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)  # Assign cluster labels
centroids = scaler.inverse_transform(kmeans.cluster_centers_)  # Transform centroids back to original scale

# Save models
joblib.dump(scaler, os.path.join(OUTPUT_DIR,'rfm_scaler.pkl'))  # Save scaler
joblib.dump(kmeans, os.path.join(OUTPUT_DIR,'kmeans_model.pkl'))  # Save KMeans model

# ============================
# PCA DIMENSION REDUCTION
# ============================
pca = PCA(n_components=2, random_state=42)  # Initialize PCA
rfm_pca = pca.fit_transform(rfm_scaled)  # Reduce scaled RFM to 2 dimensions
rfm['PCA1'], rfm['PCA2'] = rfm_pca[:,0], rfm_pca[:,1]  # Store PCA components in DataFrame

# ============================
# ASSIGN HUMAN-READABLE CLUSTER LABELS
# ============================
cluster_summary = rfm.groupby('Segment').agg(
    Recency='mean',
    Frequency='mean',
    Monetary='mean',
    CustomerID='count'
).rename(columns={'CustomerID':'Count'}).round(1)

def assign_segment_labels(summary_df):
    """
    Rank clusters and assign readable labels: High Value, Loyal, Churn Risk, Regular
    """
    labels = {}
    monetary_rank = summary_df['Monetary'].rank(method='min', ascending=False)
    recency_rank = summary_df['Recency'].rank(method='min')
    frequency_rank = summary_df['Frequency'].rank(method='min', ascending=False)
    for idx in summary_df.index:
        if monetary_rank[idx]==1:
            labels[idx]='High Value'
        elif frequency_rank[idx]==1:
            labels[idx]='Loyal'
        elif recency_rank[idx]==summary_df['Recency'].rank().max():
            labels[idx]='Churn Risk'
        else:
            labels[idx]='Regular'
    return labels

# Map labels to clusters
segment_labels = assign_segment_labels(cluster_summary)
rfm['SegmentLabel'] = rfm['Segment'].map(segment_labels)
cluster_summary['Label'] = cluster_summary.index.map(segment_labels)
cluster_summary.to_csv(os.path.join(OUTPUT_DIR,'cluster_profiles_v9.csv'),index=False)

# ============================
# DISTRIBUTION PLOTS
# ============================
def plot_distribution_with_iqr(data,column):
    """Plot histogram with Gaussian fit and IQR lines"""
    plt.figure(figsize=(7,4))
    sns.histplot(data[column],bins=30,kde=True,color='skyblue',edgecolor='black',stat='density')
    mu,std = norm.fit(data[column])
    x=np.linspace(data[column].min(),data[column].max(),100)
    plt.plot(x,norm.pdf(x,mu,std),'r--',label='Gaussian Fit')
    q1,q3 = np.percentile(data[column],[25,75])
    plt.axvline(q1,color='green',linestyle='--',label='Q1')
    plt.axvline(q3,color='purple',linestyle='--',label='Q3')
    plt.title(f'{column} Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DISTRIBUTION_DIR,f"{column}_distribution.png"),dpi=200)
    plt.close()

def plot_all_distributions(data,column):
    """Generate Full, IQR, Extremes, Top25, Bottom25 plots"""
    q1,q3 = np.percentile(data[column],[25,75])
    full = data[column]
    iqr = full[(full>=q1)&(full<=q3)]
    extremes = pd.concat([full[full<=q1],full[full>=q3]])
    top25 = full[full>=q3]
    bottom25 = full[full<=q1]

    # Full vs IQR
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    sns.histplot(full,bins=30,kde=True,color='skyblue',ax=axes[0],edgecolor='black',stat='density')
    axes[0].set_title('Full')
    sns.histplot(iqr,bins=20,kde=True,color='orange',ax=axes[1],edgecolor='black',stat='density')
    axes[1].set_title('IQR (Middle 50%)')
    plt.tight_layout()
    plt.savefig(os.path.join(DISTRIBUTION_DIR,f"{column}_iqr_comparison.png"),dpi=200)
    plt.close()

    # Extremes contrast
    fig,axes = plt.subplots(1,3,figsize=(14,4))
    sns.histplot(full,bins=30,kde=True,color='skyblue',ax=axes[0],edgecolor='black',stat='density')
    axes[0].set_title('Full')
    sns.histplot(iqr,bins=20,kde=True,color='orange',ax=axes[1],edgecolor='black',stat='density')
    axes[1].set_title('IQR')
    sns.histplot(extremes,bins=20,kde=True,color='crimson',ax=axes[2],edgecolor='black',stat='density')
    axes[2].set_title('Extremes (Top+Bottom 25%)')
    plt.tight_layout()
    plt.savefig(os.path.join(DISTRIBUTION_DIR,f"{column}_extremes_contrast.png"),dpi=200)
    plt.close()

    # Top25
    plt.figure(figsize=(6,4))
    sns.histplot(top25,bins=20,kde=True,color='red',edgecolor='black',stat='density')
    plt.title('Top25 High Spenders')
    plt.tight_layout()
    plt.savefig(os.path.join(DISTRIBUTION_DIR,f"{column}_top25.png"),dpi=200)
    plt.close()

    # Bottom25
    plt.figure(figsize=(6,4))
    sns.histplot(bottom25,bins=20,kde=True,color='blue',edgecolor='black',stat='density')
    plt.title('Bottom25 Low Value')
    plt.tight_layout()
    plt.savefig(os.path.join(DISTRIBUTION_DIR,f"{column}_bottom25.png"),dpi=200)
    plt.close()

# Apply distribution plots to all RFM features
for col in ['Recency','Frequency','Monetary']:
    plot_distribution_with_iqr(rfm,col)  # Basic distribution
    plot_all_distributions(rfm,col)  # All 5 plots

# ============================
# GAUSSIAN PROCESS REGRESSION (GPR)
# ============================
print("🔥 Training GPR model...")
X = rfm_scaled[:,:2]  # Features: Recency & Frequency
y = rfm['Monetary'].values  # Target: Monetary
kernel = C(1.0)*(RBF(1.0)) + WhiteKernel()  # Kernel definition
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)  # Instantiate GPR
gpr.fit(X,y)  # Fit model
y_pred, y_std = gpr.predict(X, return_std=True)  # Predict with uncertainty
rfm['GPR_Predicted_Monetary'] = y_pred  # Store predicted values
rfm['GPR_Uncertainty'] = y_std  # Store uncertainties
rmse = np.sqrt(mean_squared_error(y, y_pred))  # Compute RMSE
r2 = r2_score(y, y_pred)  # Compute R²
print(f"✅ GPR complete: RMSE={rmse:.2f}, R²={r2:.4f}")

# ============================
# AUTOMATED GPR SURFACE PLOTS WITH TOP25/BOTTOM25
# ============================
r = np.linspace(X[:,0].min(),X[:,0].max(),50)  # Recency grid
f = np.linspace(X[:,1].min(),X[:,1].max(),50)  # Frequency grid
R,F = np.meshgrid(r,f)  # Create meshgrid
X_grid = np.column_stack([R.ravel(), F.ravel()])  # Stack grid points
Y_mean,Y_std = gpr.predict(X_grid,return_std=True)  # Predict on grid
Y_mean = Y_mean.reshape(R.shape)  # Reshape for plotting
Y_std = Y_std.reshape(R.shape)

# Identify Top25 and Bottom25 monetary customers
q1,q3 = np.percentile(rfm['Monetary'],[25,75])
top25_mask = (rfm['Monetary'] >= q3)
bottom25_mask = (rfm['Monetary'] <= q1)

# 3D surface plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(R,F,Y_mean,cmap='viridis',alpha=0.8)  # GPR surface
ax.scatter(X[top25_mask,0],X[top25_mask,1],y[top25_mask],color='red',label='Top25',s=50)
ax.scatter(X[bottom25_mask,0],X[bottom25_mask,1],y[bottom25_mask],color='blue',label='Bottom25',s=50)
ax.set_xlabel('Recency (scaled)')
ax.set_ylabel('Frequency (scaled)')
ax.set_zlabel('Monetary')
ax.set_title('GPR Surface with Top25/Bottom25 Highlighted')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'gpr_surface_top_bottom25.png'),dpi=200)
plt.close()

# ============================
# EXPORT FINAL DATA
# ============================
rfm.to_csv(os.path.join(OUTPUT_DIR,'customers_segmented_v9.csv'),index=False)  # Save CSV
rfm.to_excel(os.path.join(OUTPUT_DIR,'customers_segmented_v9.xlsx'),index=False)  # Save Excel
print("✅ All outputs saved successfully.")
