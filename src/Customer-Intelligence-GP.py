# =============================================================================
# COMPLETE CUSTOMER SEGMENTATION & PREDICTION PIPELINE (FINAL EDITION)
# File: src/customer_segmentation_master.py
# Expects: data/Online Retail.xlsx
# Produces: timestamped outputs under outputs/outputs_YYYYMMDD_HHMM/
# Archives previous outputs under outputs/archive_YYYYMMDD_HHMM/
# Includes: RFM, KMeans (optimal k), PCA (2D+3D), GPR, full plots, exports
# Notes: All plots saved as PNG (dpi=200), console uses subtle professional colors
# =============================================================================

# -------------------------
# 0. IMPORTS
# -------------------------
import os                                           # file & directory operations
import shutil                                       # moving files & archiving
from datetime import datetime                       # timestamps for output folders
import pandas as pd                                 # DataFrame handling
import numpy as np                                  # numerical ops and arrays
import matplotlib.pyplot as plt                     # primary plotting library
import seaborn as sns                               # higher-level plotting utilities
from scipy.stats import norm                        # gaussian fit for distributions
from sklearn.preprocessing import StandardScaler    # scale features for clustering/regression
from sklearn.cluster import KMeans                  # KMeans clustering algorithm
from sklearn.metrics import silhouette_score        # silhouette metric for cluster evaluation
from sklearn.metrics import mean_squared_error, r2_score  # GPR regression metrics
from sklearn.decomposition import PCA               # PCA for dimensionality reduction
from sklearn.gaussian_process import GaussianProcessRegressor  # GPR model
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel  # kernels for GPR
import joblib                                       # save/load models to disk
from mpl_toolkits.mplot3d import Axes3D             # 3D plotting support
from colorama import init as colorama_init          # initialize colorama for colored console output
from colorama import Fore, Style                     # color constants for subtle coloring

# -------------------------
# 1. COLORAMA SETUP (console colors) - subtle professional palette
# -------------------------
colorama_init(autoreset=True)                         # initialize colorama with autoreset so colors don't persist
INFO = Fore.CYAN                                     # cyan for informational steps
SUCCESS = Fore.GREEN                                 # green for success messages
WARN = Fore.YELLOW                                   # yellow for warnings
ERR = Fore.RED                                       # red for errors

# -------------------------
# 2. PATHS, TIMESTAMPED OUTPUTS & ARCHIVAL
# -------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                       # directory containing this script
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))                    # project root (one level up)
DATA_DIR = os.path.join(BASE_DIR, 'data')                                     # data directory path
OUTPUTS_BASE = os.path.join(BASE_DIR, 'outputs')                              # base outputs directory (exact name requested)
os.makedirs(OUTPUTS_BASE, exist_ok=True)                                       # ensure outputs/ exists to allow archival

timestamp = datetime.now().strftime("%Y%m%d_%H%M")                             # create timestamp in YYYYMMDD_HHMM format
archive_dir = os.path.join(OUTPUTS_BASE, f"archive_{timestamp}")               # path for archived previous outputs
run_output_dir = os.path.join(OUTPUTS_BASE, f"outputs_{timestamp}")            # path for this run's outputs
existing_entries = [e for e in os.listdir(OUTPUTS_BASE) if e not in (os.path.basename(archive_dir), os.path.basename(run_output_dir))]  # entries to archive

if existing_entries:                                                           # if there are existing files/folders under outputs/
    os.makedirs(archive_dir, exist_ok=True)                                    # make archive directory
    print(WARN + "Archiving previous outputs to: " + archive_dir + Style.RESET_ALL)  # print archive action
    for entry in existing_entries:                                              # move each existing entry to the archive
        src = os.path.join(OUTPUTS_BASE, entry)                                 # source path to move
        dst = os.path.join(archive_dir, entry)                                  # destination inside archive
        try:
            shutil.move(src, dst)                                               # perform move
            print(INFO + f"Moved '{entry}' -> {archive_dir}" + Style.RESET_ALL) # log moved entry
        except Exception as e:
            print(ERR + f"Could not move {src} to {dst}: {e}" + Style.RESET_ALL)  # warn on failure

os.makedirs(run_output_dir, exist_ok=True)                                      # create the run-specific output folder
DISTRIBUTION_DIR = os.path.join(run_output_dir, 'distributions')                # distributions subfolder for this run
SEGMENTS_DIR = os.path.join(run_output_dir, 'segments')                         # segments folder for per-segment CSVs
os.makedirs(DISTRIBUTION_DIR, exist_ok=True)                                    # ensure distributions folder exists
os.makedirs(SEGMENTS_DIR, exist_ok=True)                                        # ensure segments folder exists

# File paths for main artifacts (within run_output_dir)
MASTER_CSV = os.path.join(run_output_dir, 'customers_segmented_master.csv')      # master CSV for Tableau
MASTER_XLSX = os.path.join(run_output_dir, 'customers_segmented_master.xlsx')    # master XLSX workbook
KMEANS_METRICS_CSV = os.path.join(run_output_dir, 'kmeans_metrics_full.csv')     # KMeans metrics CSV
CLUSTER_PROFILE_CSV = os.path.join(run_output_dir, 'cluster_profiles_full.csv')  # Cluster profiles CSV
TOP10_CSV = os.path.join(run_output_dir, 'top10_customers.csv')                  # Top10 CSV
BOTTOM10_CSV = os.path.join(run_output_dir, 'bottom10_customers.csv')            # Bottom10 CSV
CHURNED_CSV = os.path.join(run_output_dir, 'churned_customers_v2.csv')           # Churned customers CSV
ACTIVE_CSV = os.path.join(run_output_dir, 'active_customers_v2.csv')             # Active customers CSV
GPR_PKL = os.path.join(run_output_dir, 'gpr_model.pkl')                          # saved GPR model (joblib)
KMEANS_PKL = os.path.join(run_output_dir, 'kmeans_model.pkl')                    # saved KMeans model (joblib)
SCALER_PKL = os.path.join(run_output_dir, 'rfm_scaler.pkl')                      # saved scaler (joblib)

# Figure file paths
FIG_RFM_SEGMENT = os.path.join(run_output_dir, 'rfm_segmentation_tableau_ready.png')
FIG_PCA_2D = os.path.join(run_output_dir, 'pca_segmentation_2d.png')
FIG_PCA_3D = os.path.join(run_output_dir, 'pca_segmentation_3d.png')
FIG_CHURN_THRESHOLD = os.path.join(run_output_dir, 'recency_churn_threshold.png')
FIG_GPR_SURFACE = os.path.join(run_output_dir, 'gpr_predicted_surface.png')
FIG_GPR_UNCERTAINTY = os.path.join(run_output_dir, 'gpr_uncertainty_heatmap.png')
FIG_PCA_VARIANCE = os.path.join(run_output_dir, 'pca_variance_ratio.png')
FIG_ELBOW = os.path.join(run_output_dir, 'kmeans_elbow_inertia.png')
FIG_SILHOUETTE = os.path.join(run_output_dir, 'kmeans_silhouette_scores.png')
FIG_GPR_RESIDUALS = os.path.join(run_output_dir, 'gpr_actual_vs_predicted.png')
FIG_GPR_RESID_HIST = os.path.join(run_output_dir, 'gpr_residuals_histogram.png')
FIG_CLUSTER_COUNT = os.path.join(run_output_dir, 'cluster_size_bar.png')
FIG_RFM_CORR = os.path.join(run_output_dir, 'rfm_correlation_heatmap.png')

# -------------------------
# 3. DATA LOADING & VALIDATION
# -------------------------
DATA_FILE = 'Online Retail.xlsx'                          # input file name (as per your repo)
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)             # full path to dataset
if not os.path.exists(DATA_PATH):                          # check for existence
    print(ERR + f"Data file not found at {DATA_PATH}" + Style.RESET_ALL)  # error print
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")       # stop execution if missing

print(INFO + f"Loading dataset from: {DATA_PATH}" + Style.RESET_ALL)     # informational log
data = pd.read_excel(DATA_PATH)                                          # read Excel file into pandas
data.drop_duplicates(inplace=True)                                       # deduplicate rows to prevent double counting
if 'CustomerID' not in data.columns:                                     # ensure CustomerID exists
    print(ERR + "Expected column 'CustomerID' not found in dataset" + Style.RESET_ALL)  # error print
    raise KeyError("Expected column 'CustomerID' not found in dataset")  # raise informative error
data.dropna(subset=['CustomerID'], inplace=True)                         # drop rows missing CustomerID
required_cols = ['InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice']     # required columns list
for col in required_cols:                                                 # iterate required columns
    if col not in data.columns:                                           # check presence
        print(ERR + f"Expected column '{col}' not found in dataset" + Style.RESET_ALL)  # error print
        raise KeyError(f"Expected column '{col}' not found in dataset")   # raise if missing

# -------------------------
# 4. BASIC CLEANING & FEATURE CREATION
# -------------------------
data['total_spent'] = data['UnitPrice'] * data['Quantity']                # compute total spent per transaction row
data = data[data['total_spent'] > 0]                                      # keep only positive spending rows (drop refunds)
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])                 # convert InvoiceDate to datetime dtype
reference_date = data['InvoiceDate'].max()                                # choose latest invoice date as reference

# -------------------------
# 5. RFM METRICS (Recency, Frequency, Monetary)
# -------------------------
print(INFO + "Computing RFM metrics: Recency (days), Frequency (unique invoices), Monetary (sum)" + Style.RESET_ALL)  # log
rfm = data.groupby('CustomerID').agg(                                      # aggregate by customer
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),   # days since last purchase
    Frequency=('InvoiceNo', 'nunique'),                                   # number of unique invoices
    Monetary=('total_spent', 'sum')                                        # total spend per customer
).reset_index()                                                            # reset index to make CustomerID a regular column

# -------------------------
# 6. SCALE RFM FOR MODELING
# -------------------------
scaler = StandardScaler()                                                  # create scaler instance
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])  # scale RFM columns
joblib.dump(scaler, SCALER_PKL)                                            # persist scaler for reproducibility
print(SUCCESS + "RFM metrics computed and scaler saved." + Style.RESET_ALL)  # success log

# -------------------------
# 7. DETERMINE OPTIMAL K (2-10): inertia + silhouette
# -------------------------
print(INFO + "Determining optimal K for KMeans (k=2..10) using inertia & silhouette..." + Style.RESET_ALL)  # info
inertia_vals = []                                                           # list for inertia values
silhouette_vals = []                                                        # list for silhouette scores
k_candidates = list(range(2, 11))                                           # k values to evaluate
for k in k_candidates:                                                      # loop candidate k values
    km_tmp = KMeans(n_clusters=k, n_init=10, random_state=42)               # instantiate KMeans
    labels_tmp = km_tmp.fit_predict(rfm_scaled)                             # fit & predict temporary labels
    inertia_vals.append(km_tmp.inertia_)                                    # record inertia
    try:
        sil = silhouette_score(rfm_scaled, labels_tmp)                      # compute silhouette score
    except Exception:
        sil = np.nan                                                        # if silhouette fails, store NaN
    silhouette_vals.append(sil)                                             # append silhouette

metrics_df = pd.DataFrame({'k': k_candidates, 'Inertia': inertia_vals, 'Silhouette': silhouette_vals})  # compile metrics
metrics_df.to_csv(KMEANS_METRICS_CSV, index=False)                          # save metrics for auditing
print(SUCCESS + f"KMeans metrics saved to {KMEANS_METRICS_CSV}" + Style.RESET_ALL)  # success message

# Elbow (inertia) plot with legend and axis labels
plt.figure(figsize=(7, 4))                                                  # create figure for elbow
plt.plot(k_candidates, inertia_vals, marker='o', label='Inertia')           # line plot inertia vs k
plt.title('KMeans Elbow Plot (Inertia vs k)')                               # title
plt.xlabel('k (number of clusters)')                                        # x-axis label
plt.ylabel('Inertia (Sum of squared distances)')                            # y-axis label
plt.xticks(k_candidates)                                                    # set x ticks
plt.legend()                                                                # show legend
plt.tight_layout()                                                          # layout adjustment
plt.savefig(FIG_ELBOW, dpi=200)                                             # save elbow figure
plt.close()                                                                 # close figure to release memory
print(INFO + f"Saved Elbow plot to {FIG_ELBOW}" + Style.RESET_ALL)          # log save

# Silhouette plot with legend and axis labels
plt.figure(figsize=(7, 4))                                                  # create figure for silhouette
plt.plot(k_candidates, silhouette_vals, marker='o', label='Silhouette Score')  # line plot silhouette vs k
plt.title('KMeans Silhouette Score vs k')                                   # title
plt.xlabel('k (number of clusters)')                                        # x-axis label
plt.ylabel('Silhouette Score')                                              # y-axis label
plt.xticks(k_candidates)                                                    # set x ticks
plt.legend()                                                                # show legend
plt.tight_layout()                                                          # layout adjustment
plt.savefig(FIG_SILHOUETTE, dpi=200)                                        # save silhouette figure
plt.close()                                                                 # close fig
print(INFO + f"Saved Silhouette plot to {FIG_SILHOUETTE}" + Style.RESET_ALL)  # log save

# Choose optimal_k using silhouette (fallback to 3 if silhouettes NaN)
if np.all(np.isnan(silhouette_vals)):                                       # check all NaN
    optimal_k = 3                                                           # fallback
else:
    optimal_k = int(np.nanargmax(silhouette_vals) + 2)                      # choose k that maximizes silhouette (offset by 2)
print(SUCCESS + f"Optimal number of clusters selected: {optimal_k}" + Style.RESET_ALL)  # announce chosen k

# -------------------------
# 8. FINAL KMEANS FIT, ASSIGNMENT & SAVE
# -------------------------
print(INFO + "Fitting final KMeans model and assigning cluster labels..." + Style.RESET_ALL)  # info
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)            # final KMeans model
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)                              # assign cluster ids to rfm DataFrame
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)       # transform centroids back to original RFM scale
joblib.dump(kmeans, KMEANS_PKL)                                               # save KMeans model
print(SUCCESS + f"KMeans model saved to {KMEANS_PKL}" + Style.RESET_ALL)       # success log

# -------------------------
# 9. CLUSTER PROFILING & HUMAN-READABLE LABELS
# -------------------------
cluster_summary = rfm.groupby('Cluster').agg(                                 # compute cluster-level aggregates
    Recency=('Recency', 'mean'),                                              # mean recency
    Frequency=('Frequency', 'mean'),                                          # mean frequency
    Monetary=('Monetary', 'mean'),                                            # mean monetary
    Count=('CustomerID', 'count')                                             # cluster sizes
).round(1)                                                                    # round for presentation

def assign_cluster_labels(summary_df):                                        # function to assign friendly names
    labels_map = {}                                                            # label mapping dictionary
    monetary_rank = summary_df['Monetary'].rank(method='min', ascending=False) # rank monetary (1 is highest)
    recency_rank = summary_df['Recency'].rank(method='min')                    # rank recency (lower is better)
    frequency_rank = summary_df['Frequency'].rank(method='min', ascending=False)  # rank frequency (1 is highest)
    for idx in summary_df.index:                                               # loop through clusters
        if monetary_rank[idx] == 1:                                            # highest monetary cluster
            labels_map[idx] = 'High Value'                                     # label as High Value
        elif frequency_rank[idx] == 1:                                         # highest frequency cluster
            labels_map[idx] = 'Loyal'                                          # label as Loyal
        elif recency_rank[idx] == summary_df['Recency'].rank().max():          # worst recency cluster
            labels_map[idx] = 'Churn Risk'                                     # label as Churn Risk
        else:
            labels_map[idx] = 'Regular'                                        # default label
    return labels_map                                                          # return mapping

cluster_labels = assign_cluster_labels(cluster_summary)                       # compute label mapping
rfm['ClusterLabel'] = rfm['Cluster'].map(cluster_labels)                      # add friendly labels to rfm
cluster_summary['Label'] = cluster_summary.index.map(cluster_labels)          # add Label column to summary
cluster_summary.to_csv(CLUSTER_PROFILE_CSV, index=False)                      # save cluster profile CSV
print(SUCCESS + f"Cluster profiles saved to {CLUSTER_PROFILE_CSV}" + Style.RESET_ALL)  # log save

# -------------------------
# 10. TOP & BOTTOM 10 EXPORTS
# -------------------------
top10 = rfm.nlargest(10, 'Monetary')                                          # DataFrame of top 10 spenders
bottom10 = rfm.nsmallest(10, 'Monetary')                                      # DataFrame of bottom 10 spenders
top10.to_csv(TOP10_CSV, index=False)                                          # save top10 CSV
bottom10.to_csv(BOTTOM10_CSV, index=False)                                    # save bottom10 CSV
print(INFO + f"Top/bottom 10 exported to {TOP10_CSV} and {BOTTOM10_CSV}" + Style.RESET_ALL)  # log

# -------------------------
# 11. DISTRIBUTION PLOTS (IQR, Extremes, Top/Bottom with legends & axis labels)
# -------------------------
def plot_distribution_with_iqr(df, column):                                   # function for distribution + gaussian + IQR
    plt.figure(figsize=(7, 4))                                                 # figure size
    sns.histplot(df[column], bins=30, kde=True, color='skyblue', edgecolor='black', stat='density', label='Density')  # histogram w/ KDE
    mu, std = norm.fit(df[column])                                             # gaussian fit parameters
    x = np.linspace(df[column].min(), df[column].max(), 100)                   # x grid for gaussian
    plt.plot(x, norm.pdf(x, mu, std), 'r--', label='Gaussian Fit')             # gaussian overlay
    q1, q3 = np.percentile(df[column], [25, 75])                                # compute Q1 & Q3
    plt.axvline(q1, color='green', linestyle='--', label='Q1 (25%)')           # Q1 line
    plt.axvline(q3, color='purple', linestyle='--', label='Q3 (75%)')          # Q3 line
    plt.title(f'{column} Distribution with Gaussian Fit')                      # title
    plt.xlabel(f'{column} (units)')                                            # concise axis label with brackets could be added per variable below
    plt.ylabel('Density')                                                      # y label
    plt.legend()                                                               # legend for the plot
    plt.tight_layout()                                                         # adjust layout
    plt.savefig(os.path.join(DISTRIBUTION_DIR, f"{column}_distribution.png"), dpi=200)  # save figure
    plt.close()                                                                # close figure

def plot_all_distributions(df, column):                                       # function for a set of distribution comparisons
    q1, q3 = np.percentile(df[column], [25, 75])                               # quartiles
    full = df[column]                                                          # full data
    iqr = full[(full >= q1) & (full <= q3)]                                    # middle 50% subset
    extremes = pd.concat([full[full <= q1], full[full >= q3]])                 # extremes
    top25 = full[full >= q3]                                                   # top quarter
    bottom25 = full[full <= q1]                                                # bottom quarter

    # Full vs IQR (2-panel)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))                             # create subplots
    sns.histplot(full, bins=30, kde=True, color='skyblue', ax=axes[0], edgecolor='black', stat='density', label='Full')  # full
    axes[0].set_title('Full')                                                  # title left
    axes[0].set_xlabel(f'{column} (units)')                                    # x-axis label left
    axes[0].set_ylabel('Density')                                              # y label left
    sns.histplot(iqr, bins=20, kde=True, color='orange', ax=axes[1], edgecolor='black', stat='density', label='IQR')  # IQR
    axes[1].set_title('IQR (Middle 50%)')                                      # title right
    axes[1].set_xlabel(f'{column} (units)')                                    # x label right
    axes[1].set_ylabel('Density')                                              # y label right
    axes[1].legend()                                                            # legend for right panel
    axes[0].legend()                                                            # legend for left panel
    plt.tight_layout()                                                         # layout adjust
    plt.savefig(os.path.join(DISTRIBUTION_DIR, f"{column}_iqr_comparison.png"), dpi=200)  # save file
    plt.close()                                                                # close

    # Extremes vs Full (3-panel)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))                             # create 3 panels
    sns.histplot(full, bins=30, kde=True, color='skyblue', ax=axes[0], edgecolor='black', stat='density', label='Full')  # full
    axes[0].set_title('Full')                                                  # title
    axes[0].set_xlabel(f'{column} (units)')                                    # x label
    axes[0].set_ylabel('Density')                                              # y label
    sns.histplot(iqr, bins=20, kde=True, color='orange', ax=axes[1], edgecolor='black', stat='density', label='IQR')  # iqr
    axes[1].set_title('IQR')                                                   # title
    axes[1].set_xlabel(f'{column} (units)')                                    # x label
    axes[1].set_ylabel('Density')                                              # y label
    sns.histplot(extremes, bins=20, kde=True, color='crimson', ax=axes[2], edgecolor='black', stat='density', label='Extremes')  # extremes
    axes[2].set_title('Extremes (Top+Bottom 25%)')                             # title
    axes[2].set_xlabel(f'{column} (units)')                                    # x label
    axes[2].set_ylabel('Density')                                              # y label
    for ax in axes:                                                            # add legend to each axis
        ax.legend()                                                            # legend
    plt.tight_layout()                                                         # layout
    plt.savefig(os.path.join(DISTRIBUTION_DIR, f"{column}_extremes_contrast.png"), dpi=200)  # save
    plt.close()                                                                # close

    # Top 25 histogram
    plt.figure(figsize=(6, 4))                                                 # small figure
    sns.histplot(top25, bins=20, kde=True, color='red', edgecolor='black', stat='density', label='Top25')  # top quartile
    plt.title(f'{column} Top25')                                               # title
    plt.xlabel(f'{column} (units)')                                            # x label
    plt.ylabel('Density')                                                      # y label
    plt.legend()                                                               # legend
    plt.tight_layout()                                                         # layout
    plt.savefig(os.path.join(DISTRIBUTION_DIR, f"{column}_top25.png"), dpi=200)  # save
    plt.close()                                                                # close

    # Bottom 25 histogram
    plt.figure(figsize=(6, 4))                                                 # small figure
    sns.histplot(bottom25, bins=20, kde=True, color='blue', edgecolor='black', stat='density', label='Bottom25')  # bottom quartile
    plt.title(f'{column} Bottom25')                                            # title
    plt.xlabel(f'{column} (units)')                                            # x label
    plt.ylabel('Density')                                                      # y label
    plt.legend()                                                               # legend
    plt.tight_layout()                                                         # layout
    plt.savefig(os.path.join(DISTRIBUTION_DIR, f"{column}_bottom25.png"), dpi=200)  # save
    plt.close()                                                                # close

# Generate all distribution plots for Recency, Frequency, Monetary
for col in ['Recency', 'Frequency', 'Monetary']:                              # loop through RFM columns
    plot_distribution_with_iqr(rfm, col)                                      # gaussian + IQR plot
    plot_all_distributions(rfm, col)                                          # full distribution suite
print(SUCCESS + f"Distribution plots saved to {DISTRIBUTION_DIR}" + Style.RESET_ALL)  # log

# -------------------------
# 12. RECENCY-BASED CHURN THRESHOLD (Tukey & Q3) + EXPORTS
# -------------------------
print(INFO + "Calculating Recency churn thresholds (Tukey fence and Q3)..." + Style.RESET_ALL)  # info
rec_q1 = rfm['Recency'].quantile(0.25)                                         # compute Q1
rec_q3 = rfm['Recency'].quantile(0.75)                                         # compute Q3
rec_iqr = rec_q3 - rec_q1                                                      # compute IQR
rec_churn_tukey = rec_q3 + 1.5 * rec_iqr                                       # Tukey upper fence
rec_churn_q3 = rec_q3                                                           # Q3 threshold

rfm['ChurnFlag_Tukey'] = np.where(rfm['Recency'] > rec_churn_tukey, 'Churned', 'Active')  # Tukey flag
rfm['ChurnFlag_Q3'] = np.where(rfm['Recency'] > rec_churn_q3, 'Churned', 'Active')        # Q3 flag
rfm['ChurnFlag'] = rfm['ChurnFlag_Tukey']                                            # choose Tukey as primary

rfm[rfm['ChurnFlag'] == 'Churned'].to_csv(CHURNED_CSV, index=False)                # export churned list
rfm[rfm['ChurnFlag'] == 'Active'].to_csv(ACTIVE_CSV, index=False)                  # export active list

# Plot recency distribution with threshold lines, legends, axis labels
plt.figure(figsize=(8, 5))                                                       # create figure
sns.histplot(rfm['Recency'], bins=40, kde=True, color='steelblue', edgecolor='black', label='Recency')  # histogram
plt.axvline(rec_churn_tukey, color='red', linestyle='--', linewidth=2, label=f'Tukey = {rec_churn_tukey:.1f} days')  # Tukey
plt.axvline(rec_churn_q3, color='orange', linestyle='--', linewidth=1.5, label=f'Q3 = {rec_churn_q3:.1f} days')       # Q3
plt.title('Recency Distribution with Churn Thresholds')                         # title
plt.xlabel('Recency (days since last purchase)')                                # x axis label with bracket
plt.ylabel('Customer Count')                                                     # y axis label
plt.legend()                                                                     # show legend
plt.tight_layout()                                                               # layout fix
plt.savefig(FIG_CHURN_THRESHOLD, dpi=200)                                        # save figure
plt.close()                                                                      # close figure
print(SUCCESS + f"Churn thresholds applied and plot saved to {FIG_CHURN_THRESHOLD}" + Style.RESET_ALL)  # success log

# -------------------------
# 13. VALUE CATEGORIZATION & MARKETING SEGMENTS
# -------------------------
print(INFO + "Classifying customers into ValueCategory and MarketingSegment..." + Style.RESET_ALL)  # info
mon_q1 = rfm['Monetary'].quantile(0.25)                                         # monetary Q1
mon_q3 = rfm['Monetary'].quantile(0.75)                                         # monetary Q3

def classify_value(m):                                                          # function to classify monetary tiers
    if m >= mon_q3:                                                             # top quartile
        return 'High Value'                                                     # high
    elif m <= mon_q1:                                                           # bottom quartile
        return 'Low Value'                                                      # low
    else:
        return 'Medium Value'                                                   # medium

rfm['ValueCategory'] = rfm['Monetary'].apply(classify_value)                    # apply monetary classification

def marketing_segment_mapper(row):                                              # map value + churn into marketing segments
    if row['ValueCategory'] == 'High Value' and row['ChurnFlag'] == 'Active':
        return 'Champions'
    if row['ValueCategory'] == 'High Value' and row['ChurnFlag'] == 'Churned':
        return 'At-Risk High Value'
    if row['ValueCategory'] == 'Medium Value' and row['ChurnFlag'] == 'Active':
        return 'Loyal Regulars'
    if row['ValueCategory'] == 'Medium Value' and row['ChurnFlag'] == 'Churned':
        return 'At-Risk Regulars'
    if row['ValueCategory'] == 'Low Value' and row['ChurnFlag'] == 'Active':
        return 'New / Occasional'
    return 'Lost Customers'                                                      # fallback for other combinations

rfm['MarketingSegment'] = rfm.apply(marketing_segment_mapper, axis=1)           # add marketing-ready segment column

# -------------------------
# 14. PCA VISUALIZATIONS (2D + 3D) + Explained Variance
# -------------------------
print(INFO + "Running PCA (2 & 3 components) for visualization..." + Style.RESET_ALL)  # info
pca2 = PCA(n_components=2, random_state=42)                                       # 2-component PCA
rfm_pca2 = pca2.fit_transform(rfm_scaled)                                         # transform scaled RFM
rfm['PCA1'] = rfm_pca2[:, 0]                                                      # store PCA1
rfm['PCA2'] = rfm_pca2[:, 1]                                                      # store PCA2

# Explained variance bar with legend & labels
plt.figure(figsize=(6, 4))                                                        # figure for variance
explained = pca2.explained_variance_ratio_                                        # array of variance ratios
plt.bar(['PC1', 'PC2'], explained, edgecolor='black', label='Explained Variance')  # bar chart
plt.title('Explained Variance (PCA 2 components)')                                # title
plt.ylabel('Variance Ratio')                                                      # y label
plt.xlabel('Principal Components')                                                # x label
plt.legend()                                                                      # legend
plt.tight_layout()                                                                # layout
plt.savefig(FIG_PCA_VARIANCE, dpi=200)                                            # save variance figure
plt.close()                                                                       # close fig
print(INFO + f"PCA variance plot saved to {FIG_PCA_VARIANCE}" + Style.RESET_ALL)  # log

# PCA 2D scatter with legend and axis labels
plt.figure(figsize=(8, 5))                                                        # figure size
for label, color in {'High Value':'#FF6B6B','Loyal':'#4ECDC4','Regular':'#FFD93D','Churn Risk':'#1A535C'}.items():  # color palette
    subset = rfm[rfm['ClusterLabel'] == label]                                    # subset by friendly label
    plt.scatter(subset['PCA1'], subset['PCA2'], label=label, color=color, alpha=0.8, edgecolor='k', s=60)  # plot points
plt.title('Customer Segments (PCA 2D)')                                            # title
plt.xlabel('PCA1 (Customer Value Dimension)')                                      # x label with bracketed meaning
plt.ylabel('PCA2 (Engagement Dimension)')                                          # y label
plt.legend(title='Segment')                                                        # legend with title
plt.tight_layout()                                                                 # layout
plt.savefig(FIG_PCA_2D, dpi=200)                                                   # save 2D PCA plot
plt.close()                                                                        # close figure
print(INFO + f"PCA 2D plot saved to {FIG_PCA_2D}" + Style.RESET_ALL)               # log

# PCA 3D
pca3 = PCA(n_components=3, random_state=42)                                        # 3-component PCA
rfm_pca3 = pca3.fit_transform(rfm_scaled)                                          # transform scaled data
rfm['PCA3_1'] = rfm_pca3[:, 0]                                                     # store first 3D component
rfm['PCA3_2'] = rfm_pca3[:, 1]                                                     # store second 3D component
rfm['PCA3_3'] = rfm_pca3[:, 2]                                                     # store third 3D component

fig = plt.figure(figsize=(8, 6))                                                   # figure for 3D PCA
ax = fig.add_subplot(111, projection='3d')                                         # 3D axes
for label, color in {'High Value':'#FF6B6B','Loyal':'#4ECDC4','Regular':'#FFD93D','Churn Risk':'#1A535C'}.items():  # palette
    subset = rfm[rfm['ClusterLabel'] == label]                                     # subset
    ax.scatter(subset['PCA3_1'], subset['PCA3_2'], subset['PCA3_3'], label=label, color=color, s=40, alpha=0.7, edgecolor='k')  # 3D scatter
ax.set_title('Customer Segments (PCA 3D)')                                          # title
ax.set_xlabel('PCA1')                                                              # x label
ax.set_ylabel('PCA2')                                                              # y label
ax.set_zlabel('PCA3')                                                              # z label
ax.legend(title='Segment')                                                         # legend
plt.tight_layout()                                                                 # layout
plt.savefig(FIG_PCA_3D, dpi=200)                                                   # save 3D PCA
plt.close()                                                                        # close figure
print(INFO + f"PCA 3D plot saved to {FIG_PCA_3D}" + Style.RESET_ALL)               # log

# -------------------------
# 15. RFM SCATTER (Recency vs Monetary) - Tableau-ready with centroid overlay
# -------------------------
plt.figure(figsize=(8, 5))                                                         # figure for RFM scatter
for label, color in {'High Value':'#FF6B6B','Loyal':'#4ECDC4','Regular':'#FFD93D','Churn Risk':'#1A535C'}.items():  # palette
    subset = rfm[rfm['ClusterLabel'] == label]                                     # subset rows
    plt.scatter(subset['Recency'], subset['Monetary'], label=label, color=color, alpha=0.7, edgecolor='k', s=60)  # scatter plot
centroids_df = pd.DataFrame(centroids_original, columns=['Recency','Frequency','Monetary'])  # centroids in original RFM scale
plt.scatter(centroids_df['Recency'], centroids_df['Monetary'], color='red', marker='X', s=200, label='Centroids')  # plot centroids
plt.title('Customer Segments (RFM)')                                               # title
plt.xlabel('Recency (days since last purchase)')                                   # x label with unit meaning
plt.ylabel('Monetary (total spend)')                                               # y label with unit meaning
plt.legend(title='Segment', bbox_to_anchor=(1.05,1), loc='upper left')             # legend outside plot
plt.tight_layout()                                                                 # layout
plt.savefig(FIG_RFM_SEGMENT, dpi=200)                                              # save RFM scatter
plt.close()                                                                        # close figure
print(INFO + f"RFM segmentation scatter saved to {FIG_RFM_SEGMENT}" + Style.RESET_ALL)  # log

# -------------------------
# 16. CLUSTER COUNT BAR PLOT (new optional diagnostic)
# -------------------------
cluster_counts = rfm['ClusterLabel'].value_counts().sort_index()                   # compute counts per friendly label
plt.figure(figsize=(7, 5))                                                         # figure for cluster bar chart
cluster_counts.plot(kind='bar', color=['#FF6B6B','#4ECDC4','#FFD93D','#1A535C'], edgecolor='black')  # bar chart with palette
plt.title('Cluster Size Distribution')                                             # title
plt.xlabel('Cluster (Label)')                                                      # x label
plt.ylabel('Number of Customers')                                                  # y label
plt.xticks(rotation=45)                                                            # rotate x labels for readability
plt.tight_layout()                                                                 # layout
plt.legend(['Customer Count'], loc='upper right')                                  # legend
plt.savefig(FIG_CLUSTER_COUNT, dpi=200)                                            # save cluster count plot
plt.close()                                                                        # close figure
print(INFO + f"Cluster size bar plot saved to {FIG_CLUSTER_COUNT}" + Style.RESET_ALL)  # log

# -------------------------
# 17. RFM CORRELATION HEATMAP (new optional diagnostic)
# -------------------------
plt.figure(figsize=(6, 5))                                                         # figure for correlation heatmap
corr = rfm[['Recency','Frequency','Monetary']].corr()                              # compute correlation matrix
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)         # heatmap with annotations
plt.title('RFM Correlation Heatmap')                                               # title
plt.tight_layout()                                                                 # layout
plt.savefig(FIG_RFM_CORR, dpi=200)                                                 # save correlation heatmap
plt.close()                                                                        # close figure
print(INFO + f"RFM correlation heatmap saved to {FIG_RFM_CORR}" + Style.RESET_ALL)  # log

# -------------------------
# 18. GAUSSIAN PROCESS REGRESSION (GPR) FOR MONETARY PREDICTION
# -------------------------
print(INFO + "Training Gaussian Process Regression (GPR) model..." + Style.RESET_ALL)  # info
X = rfm_scaled[:, :2]                                                              # features: scaled Recency & Frequency
y = rfm['Monetary'].values                                                         # target: Monetary

kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2,1e2)) + WhiteKernel()  # kernel setup
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=3, random_state=42)  # instantiate GPR
gpr.fit(X, y)                                                                       # fit model on training data
joblib.dump(gpr, GPR_PKL)                                                           # persist GPR model
print(SUCCESS + f"GPR model trained and saved to {GPR_PKL}" + Style.RESET_ALL)       # success log

# Predictions and uncertainty on training set
y_pred, y_std = gpr.predict(X, return_std=True)                                    # mean prediction & std
rfm['GPR_Predicted_Monetary'] = y_pred                                              # attach predicted monetary to rfm
rfm['GPR_Uncertainty'] = y_std                                                       # attach uncertainty to rfm
gpr_rmse = np.sqrt(mean_squared_error(y, y_pred))                                   # compute RMSE
gpr_r2 = r2_score(y, y_pred)                                                         # compute R^2
print(SUCCESS + f"GPR performance -> RMSE: {gpr_rmse:.2f}, R²: {gpr_r2:.4f}" + Style.RESET_ALL)  # print metrics

# GPR predicted surface (3D)
r_vals = np.linspace(X[:,0].min(), X[:,0].max(), 50)                                # Recency grid (scaled)
f_vals = np.linspace(X[:,1].min(), X[:,1].max(), 50)                                # Frequency grid (scaled)
R_grid, F_grid = np.meshgrid(r_vals, f_vals)                                        # meshgrid for plotting
X_grid = np.column_stack([R_grid.ravel(), F_grid.ravel()])                          # flatten grid for prediction
Y_mean, Y_std_grid = gpr.predict(X_grid, return_std=True)                            # predict on grid
Y_mean = Y_mean.reshape(R_grid.shape)                                                # reshape mean to grid
Y_std_grid = Y_std_grid.reshape(R_grid.shape)                                        # reshape std to grid

fig = plt.figure(figsize=(8,6))                                                      # figure for GPR surface
ax = fig.add_subplot(111, projection='3d')                                          # 3D axis
ax.plot_surface(R_grid, F_grid, Y_mean, cmap='viridis', alpha=0.8)                   # plot surface
ax.set_xlabel('Recency (scaled)')                                                    # x label
ax.set_ylabel('Frequency (scaled)')                                                  # y label
ax.set_zlabel('Predicted Monetary')                                                  # z label
ax.set_title('GPR Predicted Monetary Surface')                                       # title
plt.tight_layout()                                                                   # layout
plt.savefig(FIG_GPR_SURFACE, dpi=200)                                                # save GPR surface
plt.close()                                                                          # close figure
print(INFO + f"GPR surface plot saved to {FIG_GPR_SURFACE}" + Style.RESET_ALL)       # log

# GPR uncertainty heatmap (already included earlier but saved here too)
plt.figure(figsize=(6,5))                                                            # figure for uncertainty heatmap
plt.contourf(R_grid, F_grid, Y_std_grid, levels=20, cmap='coolwarm')                 # filled contour plot
plt.colorbar(label='Prediction Std. Dev. (Uncertainty)')                              # colorbar with label
plt.xlabel('Recency (scaled)')                                                       # x label
plt.ylabel('Frequency (scaled)')                                                     # y label
plt.title('GPR Prediction Uncertainty')                                              # title
plt.tight_layout()                                                                   # layout
plt.savefig(FIG_GPR_UNCERTAINTY, dpi=200)                                            # save heatmap
plt.close()                                                                          # close figure
print(INFO + f"GPR uncertainty heatmap saved to {FIG_GPR_UNCERTAINTY}" + Style.RESET_ALL)  # log

# -------------------------
# 19. GPR DIAGNOSTICS: ACTUAL vs PREDICTED (scatter) & RESIDUALS HISTOGRAM
# -------------------------
plt.figure(figsize=(7,6))                                                            # figure for actual vs predicted scatter
plt.scatter(y, y_pred, alpha=0.7, edgecolor='k', s=50, label='Customers')           # scatter actual vs predicted
max_val = max(np.nanmax(y), np.nanmax(y_pred))                                       # compute max for identity line
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Identity (y=x)')    # identity line
plt.xlabel('Actual Monetary')                                                        # x label
plt.ylabel('Predicted Monetary (GPR)')                                               # y label
plt.title('GPR: Actual vs Predicted Monetary')                                       # title
plt.legend()                                                                         # legend
plt.tight_layout()                                                                   # layout
plt.savefig(FIG_GPR_RESIDUALS, dpi=200)                                              # save scatter
plt.close()                                                                          # close figure
print(INFO + f"GPR Actual vs Predicted scatter saved to {FIG_GPR_RESIDUALS}" + Style.RESET_ALL)  # log

# Residuals histogram
residuals = y - y_pred                                                               # compute residuals
plt.figure(figsize=(7,5))                                                            # figure for residuals
sns.histplot(residuals, bins=40, kde=True, color='teal', edgecolor='black', label='Residuals')  # histogram with KDE
plt.axvline(np.mean(residuals), color='red', linestyle='--', label=f'Mean Residual = {np.mean(residuals):.2f}')  # mean line
plt.title('GPR Residuals Distribution (Actual - Predicted)')                         # title
plt.xlabel('Residual (Actual - Predicted)')                                          # x label
plt.ylabel('Count')                                                                   # y label
plt.legend()                                                                         # legend
plt.tight_layout()                                                                   # layout
plt.savefig(FIG_GPR_RESID_HIST, dpi=200)                                             # save residuals histogram
plt.close()                                                                          # close figure
print(INFO + f"GPR residuals histogram saved to {FIG_GPR_RESID_HIST}" + Style.RESET_ALL)  # log

# -------------------------
# 20. EXPORT ALL DATA, SEGMENTS & WORKBOOK
# -------------------------
rfm.to_csv(MASTER_CSV, index=False)                                                  # export master CSV for Tableau ingestion
rfm.to_excel(MASTER_XLSX, index=False)                                               # export master XLSX workbook
print(SUCCESS + f"Master CSV & XLSX exported to {run_output_dir}" + Style.RESET_ALL)  # log

# Export per-marketing-segment CSVs for marketing automation use
for seg_name, seg_df in rfm.groupby('MarketingSegment'):                             # iterate marketing segments
    safe_name = seg_name.replace('/', '_').replace(' ', '_')                        # sanitize filename
    seg_path = os.path.join(SEGMENTS_DIR, f"{safe_name}.csv")                        # segment CSV path
    seg_df.to_csv(seg_path, index=False)                                             # save segment CSV
print(INFO + f"Per-segment CSVs saved to {SEGMENTS_DIR}" + Style.RESET_ALL)          # log

# Export segment summary CSV
segment_summary = rfm.groupby('MarketingSegment').agg(                               # aggregate by marketing segment
    CustomerCount=('CustomerID', 'count'),
    AvgRecency=('Recency', 'mean'),
    AvgFrequency=('Frequency', 'mean'),
    AvgMonetary=('Monetary', 'mean'),
    TotalMonetary=('Monetary', 'sum')
).sort_values(by='TotalMonetary', ascending=False).reset_index()                     # sort by revenue contribution
segment_summary.to_csv(os.path.join(run_output_dir, 'RFM_Marketing_Segment_Summary.csv'), index=False)  # save summary
print(INFO + f"Marketing segment summary saved to {os.path.join(run_output_dir, 'RFM_Marketing_Segment_Summary.csv')}" + Style.RESET_ALL)  # log

# Ensure cluster_summary saved (redundant save to run folder)
cluster_summary.to_csv(os.path.join(run_output_dir, 'cluster_profiles_full.csv'), index=False)  # save cluster profiles again

# Save top & bottom lists into run folder as well for convenience
top10.to_csv(os.path.join(run_output_dir, 'top10_customers.csv'), index=False)        # save top10
bottom10.to_csv(os.path.join(run_output_dir, 'bottom10_customers.csv'), index=False)  # save bottom10

# Compose an Excel workbook containing important sheets for stakeholders
with pd.ExcelWriter(os.path.join(run_output_dir, 'RFM_Segmentation_Workbook.xlsx'), engine='openpyxl') as writer:
    rfm.to_excel(writer, sheet_name='Master', index=False)                            # master sheet
    segment_summary.to_excel(writer, sheet_name='Segment_Summary', index=False)       # marketing segment summary
    cluster_summary.to_excel(writer, sheet_name='Cluster_Profiles', index=False)      # cluster profiles
    top10.to_excel(writer, sheet_name='Top_10', index=False)                          # top10 sheet
    bottom10.to_excel(writer, sheet_name='Bottom_10', index=False)                    # bottom10 sheet
    for seg_name, seg_df in rfm.groupby('MarketingSegment'):                          # add each marketing segment as a sheet
        safe_sheet = seg_name[:31].replace('/', '_').replace(' ', '_')                # sheet name sanitized to 31 chars
        seg_df.to_excel(writer, sheet_name=safe_sheet, index=False)                  # write segment sheet
print(SUCCESS + f"Excel workbook created at {os.path.join(run_output_dir, 'RFM_Segmentation_Workbook.xlsx')}" + Style.RESET_ALL)  # log

# -------------------------
# 21. SAVE MODEL ARTIFACTS (ensure saved)
# -------------------------
joblib.dump(scaler, SCALER_PKL)                                                     # save scaler again
joblib.dump(kmeans, KMEANS_PKL)                                                     # save kmeans again
joblib.dump(gpr, GPR_PKL)                                                           # save gpr again
print(SUCCESS + "Model artifacts saved (scaler, kmeans, gpr)." + Style.RESET_ALL)   # log

# -------------------------
# 22. FINAL SUMMARY TO CONSOLE
# -------------------------
print("\n" + SUCCESS + "FINAL MASTER PIPELINE RUN COMPLETE" + Style.RESET_ALL)       # completion banner
print(INFO + f"Run timestamp: {timestamp}" + Style.RESET_ALL)                       # timestamp info
print(INFO + f"Run output folder: {run_output_dir}" + Style.RESET_ALL)              # run output folder path
print(INFO + f"Archive folder: {archive_dir if existing_entries else '(none)'}" + Style.RESET_ALL)  # archive folder or none
print(INFO + f"Master CSV: {MASTER_CSV}" + Style.RESET_ALL)                         # master CSV path
print(INFO + f"Master XLSX: {MASTER_XLSX}" + Style.RESET_ALL)                       # master XLSX path
print(INFO + f"Cluster profiles: {CLUSTER_PROFILE_CSV}" + Style.RESET_ALL)          # cluster profiles CSV
print(INFO + f"Top10/Bottom10: {TOP10_CSV} / {BOTTOM10_CSV}" + Style.RESET_ALL)     # top/bottom paths
print(INFO + f"Churn lists: {CHURNED_CSV} / {ACTIVE_CSV}" + Style.RESET_ALL)        # churn lists
print(INFO + f"PCA figures: {FIG_PCA_2D}, {FIG_PCA_3D}, variance: {FIG_PCA_VARIANCE}" + Style.RESET_ALL)  # PCA figs
print(INFO + f"GPR figures: {FIG_GPR_SURFACE}, {FIG_GPR_UNCERTAINTY}, {FIG_GPR_RESIDUALS}, {FIG_GPR_RESID_HIST}" + Style.RESET_ALL)  # GPR figs
print(INFO + f"Distribution plots folder: {DISTRIBUTION_DIR}" + Style.RESET_ALL)     # distributions folder
print(INFO + f"Saved models: scaler -> {SCALER_PKL}, kmeans -> {KMEANS_PKL}, gpr -> {GPR_PKL}" + Style.RESET_ALL)  # saved models
print(INFO + f"Total customers analyzed: {len(rfm)}" + Style.RESET_ALL)              # number of customers processed
print(INFO + "Point Tableau at the MASTER_CSV in the run output folder to visualize results." + Style.RESET_ALL)  # next step guidance

# =============================================================================
# END OF SCRIPT - All requested plots and features included, with colored console
# =============================================================================
