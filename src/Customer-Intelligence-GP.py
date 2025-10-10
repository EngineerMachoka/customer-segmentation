# ===================================================================
# COMBINED MASTER: RFM + KMEANS CLUSTERING + DISTRIBUTIONS + AUTO-CHURN
# + VALUE SEGMENTS + MARKETING SEGMENTS + EXPORTS (CSV + Excel)
# ===================================================================

# ---------------------------
# 0. IMPORTS (all used libraries)
# ---------------------------
import os                                                                              # file and directory operations
import random                                                                          # random integers used for renaming backup files
import pandas as pd                                                                    # data manipulation with DataFrame
import numpy as np                                                                     # numerical operations
import matplotlib.pyplot as plt                                                        # plotting
import seaborn as sns                                                                  # statistical plotting
from sklearn.preprocessing import StandardScaler                                      # scaling for clustering
from sklearn.cluster import KMeans                                                     # k-means clustering
from sklearn.metrics import silhouette_score                                          # silhouette metric for cluster quality
from sklearn.decomposition import PCA                                                 # principal component analysis for visualization
from scipy.stats import norm                                                           # gaussian fitting for distribution plots
import joblib                                                                          # save/load fitted models
from mpl_toolkits.mplot3d import Axes3D                                               # optional 3D plotting support
import openpyxl                                                                        # excel writer engine (used by pandas ExcelWriter)
from datetime import timedelta                                                         # helper for snapshot timedelta

# ---------------------------
# 1. CONFIGURATION PATHS & FOLDERS
# ---------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                               # directory of this script file
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))                             # project root (one level up) - adapt if needed
DATA_DIR = os.path.join(BASE_DIR, 'data')                                              # folder for input data files
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')                                         # folder for outputs (csvs, models, images)
DISTRIBUTION_DIR = os.path.join(OUTPUT_DIR, 'distributions')                           # folder specifically for distribution plots
os.makedirs(DATA_DIR, exist_ok=True)                                                   # ensure data folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)                                                 # ensure outputs folder exists
os.makedirs(DISTRIBUTION_DIR, exist_ok=True)                                           # ensure distributions folder exists

# ---------------------------
# 2. RENAME EXISTING FILES IN OUTPUT (avoid overwrite)
# ---------------------------
def rename_existing_files(folder):                                                      # function to rename files to avoid overwriting previous outputs
    for root, dirs, files in os.walk(folder):                                          # walk through folder structure
        for filename in files:                                                         # iterate file names
            old_path = os.path.join(root, filename)                                    # full path to existing file
            rand_prefix = str(random.randint(1000, 9999))                              # create random numeric prefix
            new_filename = f"{rand_prefix}_{filename}"                                 # build new filename with prefix
            new_path = os.path.join(root, new_filename)                                # new full path
            try:
                os.rename(old_path, new_path)                                          # attempt rename
                print(f"📂 Renamed existing file → {new_filename}")                     # log rename success
            except Exception as e:                                                      # catch any error during rename
                print(f"⚠️ Could not rename {filename}: {e}")                           # log error if rename fails

rename_existing_files(OUTPUT_DIR)                                                       # run file rename to prevent accidental overwrites

# ---------------------------
# 3. LOAD DATA
# ---------------------------
DATA_FILE = 'Online Retail.xlsx'                                                       # expected Excel file name (update if needed)
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)                                          # full path to data file
if not os.path.exists(DATA_PATH):                                                       # check if file exists
    raise FileNotFoundError(f"❌ Data file not found: {DATA_PATH}")                     # raise informative error if missing

print("📦 Loading dataset...")                                                           # notify user loading starts
data = pd.read_excel(DATA_PATH)                                                         # load dataset from Excel into pandas DataFrame
data.drop_duplicates(inplace=True)                                                      # drop exact duplicate rows to clean dataset
if 'CustomerID' not in data.columns:                                                   # validate expected columns exist
    raise KeyError("❌ Expected column 'CustomerID' not found in data")                 # raise descriptive error if missing CustomerID
data.dropna(subset=['CustomerID'], inplace=True)                                       # drop rows without CustomerID (we need customers)
# normalize column names (optional): not modifying here to preserve original names

# ---------------------------
# 4. BASIC CLEANING & FEATURE CREATION
# ---------------------------
# compute transaction-level total (monetary) and filter invalid values
if 'Quantity' not in data.columns or 'UnitPrice' not in data.columns:                  # confirm required columns
    raise KeyError("❌ Expected 'Quantity' and 'UnitPrice' columns in the dataset")     # error if missing critical columns
data['total_spent'] = data['UnitPrice'] * data['Quantity']                             # compute transaction-level total value
data = data[data['total_spent'] > 0]                                                    # remove refunds or zero/negative totals
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])                               # ensure InvoiceDate is datetime dtype
reference_date = data['InvoiceDate'].max()                                              # set reference date for recency calculations

# ---------------------------
# 5. COMPUTE RFM METRICS
# ---------------------------
print("🧮 Computing RFM metrics...")                                                      # notify RFM computation
rfm = data.groupby('CustomerID').agg(                                                   # aggregate by CustomerID
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),                 # Recency: days since last purchase
    Frequency=('InvoiceNo', 'nunique'),                                                 # Frequency: number of unique invoices
    Monetary=('total_spent', 'sum')                                                     # Monetary: total spending per customer
).reset_index()                                                                         # reset index to make CustomerID a column

# ---------------------------
# 6. SCALE RFM FOR CLUSTERING
# ---------------------------
scaler = StandardScaler()                                                               # instantiate standard scaler
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])            # fit scaler and transform RFM features
# save scaler for future inverse transforms / pipelines
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'rfm_scaler.pkl'))                         # persist scaler to outputs

# ---------------------------
# 7. DETERMINE OPTIMAL K FOR KMEANS (inertia + silhouette)
# ---------------------------
print("🔍 Determining optimal cluster count...")                                        # notify start of clustering selection
inertia = []                                                                            # list to store inertia values
silhouette_scores = []                                                                  # list to store silhouette scores
k_range = range(2, 11)                                                                  # search k from 2 to 10 inclusive
for k in k_range:                                                                       # iterate candidate cluster counts
    km = KMeans(n_clusters=k, n_init=10, random_state=42)                               # create KMeans with fixed random state for reproducibility
    labels = km.fit_predict(rfm_scaled)                                                 # fit and predict cluster labels
    inertia.append(km.inertia_)                                                         # collect inertia (sum of squared distances)
    try:
        s = silhouette_score(rfm_scaled, labels)                                       # compute silhouette score for cluster validity
    except Exception:
        s = np.nan                                                                      # if silhouette cannot be computed, record NaN
    silhouette_scores.append(s)                                                         # append silhouette score to list

# create metrics DataFrame and export
metrics_df = pd.DataFrame({'k': list(k_range), 'Inertia': inertia, 'Silhouette': silhouette_scores})  # metrics dataframe
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'kmeans_metrics_full.csv'), index=False)     # export metrics for posterity
# choose optimal_k based on silhouette (highest silhouette) falling back to elbow (inertia) if NaNs present
if np.all(np.isnan(silhouette_scores)):                                                  # if all silhouette scores are NaN
    optimal_k = 3                                                                       # fallback default k
else:
    optimal_k = int(np.nanargmax(silhouette_scores) + 2)                                # pick k corresponding to max silhouette (offset +2 because k_range starts at 2)

print(f"✅ Optimal clusters determined: {optimal_k}")                                     # log chosen k

# ---------------------------
# 8. FINAL KMEANS CLUSTERING (apply with optimal_k)
# ---------------------------
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)                       # instantiate final kmeans model
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)                                         # assign cluster labels to rfm table
centroids_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)            # convert centroids back to original RFM scale
joblib.dump(kmeans, os.path.join(OUTPUT_DIR, 'kmeans_model.pkl'))                        # persist kmeans model for reuse

# ---------------------------
# 9. PCA FOR 2D VISUALIZATION
# ---------------------------
pca = PCA(n_components=2, random_state=42)                                              # create PCA reducer
rfm_pca = pca.fit_transform(rfm_scaled)                                                 # transform scaled RFM into 2 principal components
rfm['PCA1'] = rfm_pca[:, 0]                                                             # store PCA1
rfm['PCA2'] = rfm_pca[:, 1]                                                             # store PCA2

# save a basic PCA scatter plot colored by cluster
plt.figure(figsize=(8, 6))                                                              # set figure size for PCA scatter
sns.scatterplot(x='PCA1', y='PCA2', hue='Segment', data=rfm, palette='tab10', legend='full')  # scatter colored by cluster
plt.title('PCA Projection of RFM Clusters')                                             # title for plot
plt.tight_layout()                                                                      # adjust layout
plt.savefig(os.path.join(OUTPUT_DIR, 'rfm_pca_clusters.png'), dpi=200)                  # save PCA scatter image
plt.close()                                                                             # close figure to free memory

# ---------------------------
# 10. CLUSTER SUMMARY & HUMAN-READABLE LABELS
# ---------------------------
cluster_summary = rfm.groupby('Segment').agg(                                           # aggregate cluster statistics
    Recency=('Recency', 'mean'),                                                        # mean recency per cluster
    Frequency=('Frequency', 'mean'),                                                    # mean frequency per cluster
    Monetary=('Monetary', 'mean'),                                                      # mean monetary per cluster
    Count=('CustomerID', 'count')                                                       # number of customers per cluster
).round(1)                                                                              # round numeric results to 1 decimal place

# create human readable labels for clusters using ranks
def assign_segment_labels(summary_df):                                                  # function maps numeric cluster properties to friendly names
    labels = {}                                                                         # dictionary to hold mapping
    monetary_rank = summary_df['Monetary'].rank(method='min', ascending=False)         # rank clusters by monetary desc
    recency_rank = summary_df['Recency'].rank(method='min')                            # rank clusters by recency asc (lower recency = more recent)
    frequency_rank = summary_df['Frequency'].rank(method='min', ascending=False)       # rank clusters by frequency desc
    for idx in summary_df.index:                                                        # iterate cluster indices
        if monetary_rank[idx] == 1:                                                     # top monetary cluster
            labels[idx] = 'High Value'                                                  # label high value
        elif frequency_rank[idx] == 1:                                                  # top frequency cluster
            labels[idx] = 'Loyal'                                                       # label loyal
        elif recency_rank[idx] == summary_df['Recency'].rank().max():                  # highest recency (most stale)
            labels[idx] = 'Churn Risk'                                                  # mark churn risk cluster
        else:
            labels[idx] = 'Regular'                                                     # default label
    return labels                                                                       # return mapping

segment_labels = assign_segment_labels(cluster_summary)                                # get mapping of cluster->label
rfm['SegmentLabel'] = rfm['Segment'].map(segment_labels)                               # map labels back to rfm
cluster_summary['Label'] = cluster_summary.index.map(segment_labels)                    # include Label in summary
cluster_summary.to_csv(os.path.join(OUTPUT_DIR, 'cluster_profiles_full.csv'), index=False)  # export cluster summary

# ---------------------------
# 11. DISTRIBUTION PLOTTING FUNCTIONS (detailed)
# ---------------------------
def plot_distribution_with_iqr(dataframe, column):                                     # function draws distribution and marks IQR with gaussian fit
    plt.figure(figsize=(7, 4))                                                          # create figure with size
    sns.histplot(dataframe[column], bins=30, kde=True, color='skyblue', edgecolor='black', stat='density')  # histogram as density with kde overlay
    mu, std = norm.fit(dataframe[column])                                               # fit normal distribution to data
    x = np.linspace(dataframe[column].min(), dataframe[column].max(), 100)              # x-values for Gaussian overlay
    plt.plot(x, norm.pdf(x, mu, std), 'r--', label='Gaussian Fit')                     # plot Gaussian fit as dashed red line
    q1, q3 = np.percentile(dataframe[column], [25, 75])                                 # compute quartiles
    plt.axvline(q1, color='green', linestyle='--', label='Q1')                         # mark Q1
    plt.axvline(q3, color='purple', linestyle='--', label='Q3')                        # mark Q3
    plt.title(f'{column} Distribution')                                                 # title for the plot
    plt.legend()                                                                        # show legend
    plt.tight_layout()                                                                  # adjust layout
    plt.savefig(os.path.join(DISTRIBUTION_DIR, f"{column}_distribution.png"), dpi=200)  # save image to distributions folder
    plt.close()                                                                         # close to free memory

def plot_all_distributions(dataframe, column):                                         # function generates a suite of distribution plots for a column
    q1, q3 = np.percentile(dataframe[column], [25, 75])                                 # compute first and third quartiles
    full = dataframe[column]                                                            # full series values
    iqr = full[(full >= q1) & (full <= q3)]                                             # middle 50% values
    extremes = pd.concat([full[full <= q1], full[full >= q3]])                          # extremes (bottom 25% + top 25%)
    top25 = full[full >= q3]                                                            # top quartile
    bottom25 = full[full <= q1]                                                         # bottom quartile

    # Full vs IQR comparison (two panels)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))                                     # create two subplot axes
    sns.histplot(full, bins=30, kde=True, color='skyblue', ax=axes[0], edgecolor='black', stat='density')  # full distribution plot
    axes[0].set_title('Full')                                                           # title left subplot
    sns.histplot(iqr, bins=20, kde=True, color='orange', ax=axes[1], edgecolor='black', stat='density')     # IQR distribution plot
    axes[1].set_title('IQR (Middle 50%)')                                               # title right subplot
    plt.tight_layout()                                                                  # adjust layout
    plt.savefig(os.path.join(DISTRIBUTION_DIR, f"{column}_iqr_comparison.png"), dpi=200)  # save combined image
    plt.close()                                                                         # close figure

    # Extremes vs Full (three panels)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))                                     # create three subplot axes
    sns.histplot(full, bins=30, kde=True, color='skyblue', ax=axes[0], edgecolor='black', stat='density')  # full
    axes[0].set_title('Full')                                                           # title
    sns.histplot(iqr, bins=20, kde=True, color='orange', ax=axes[1], edgecolor='black', stat='density')     # iqr
    axes[1].set_title('IQR')                                                            # title
    sns.histplot(extremes, bins=20, kde=True, color='crimson', ax=axes[2], edgecolor='black', stat='density')  # extremes
    axes[2].set_title('Extremes (Top+Bottom 25%)')                                      # title
    plt.tight_layout()                                                                  # layout adjust
    plt.savefig(os.path.join(DISTRIBUTION_DIR, f"{column}_extremes_contrast.png"), dpi=200)  # save image
    plt.close()                                                                         # close

    # Top25 only
    plt.figure(figsize=(6, 4))                                                          # new figure for top25
    sns.histplot(top25, bins=20, kde=True, color='red', edgecolor='black', stat='density')  # plot top25
    plt.title('Top25 High Spenders')                                                   # title
    plt.tight_layout()                                                                  # layout adjust
    plt.savefig(os.path.join(DISTRIBUTION_DIR, f"{column}_top25.png"), dpi=200)        # save
    plt.close()                                                                         # close

    # Bottom25 only
    plt.figure(figsize=(6, 4))                                                          # new figure for bottom25
    sns.histplot(bottom25, bins=20, kde=True, color='blue', edgecolor='black', stat='density')  # plot bottom25
    plt.title('Bottom25 Low Value')                                                    # title
    plt.tight_layout()                                                                  # layout adjust
    plt.savefig(os.path.join(DISTRIBUTION_DIR, f"{column}_bottom25.png"), dpi=200)     # save
    plt.close()                                                                         # close

# ---------------------------
# 12. GENERATE DISTRIBUTION PLOTS FOR R, F, M
# ---------------------------
for col in ['Recency', 'Frequency', 'Monetary']:                                        # iterate desired RFM columns
    plot_distribution_with_iqr(rfm, col)                                                # create single distribution plot with IQR lines
    plot_all_distributions(rfm, col)                                                    # create the full suite of distribution plots

# ---------------------------
# 13. AUTOMATIC CHURN DETECTION (DATA-DRIVEN) & CLASSIFICATION
# ---------------------------
# We will reuse Recency distribution to mark churn automatically using IQR-based cutoff

# compute quartiles and IQR for Recency (already computed earlier for clustering; recomputing for clarity)
rec_q1 = rfm['Recency'].quantile(0.25)                                                  # recency 25th percentile
rec_q3 = rfm['Recency'].quantile(0.75)                                                  # recency 75th percentile
rec_iqr = rec_q3 - rec_q1                                                               # recency interquartile range
rec_churn_threshold = rec_q3 + 1.5 * rec_iqr                                            # churn threshold defined as Q3 + 1.5*IQR
rfm['ChurnStatus'] = np.where(rfm['Recency'] > rec_churn_threshold, 'Churned', 'Active')  # label churn based on threshold
print(f"📊 Automatic Recency churn threshold (days): {rec_churn_threshold:.2f}")         # log churn threshold

# ---------------------------
# 14. VALUE CATEGORIZATION (Monetary quartile-based)
# ---------------------------
mon_q1 = rfm['Monetary'].quantile(0.25)                                                 # monetary quartile 1
mon_q3 = rfm['Monetary'].quantile(0.75)                                                 # monetary quartile 3

def classify_value_tiers(monetary_value):                                               # function to return value tier label
    if monetary_value >= mon_q3:                                                         # top 25%
        return 'High Value'                                                              # assign high value
    elif monetary_value <= mon_q1:                                                       # bottom 25%
        return 'Low Value'                                                               # assign low value
    else:
        return 'Medium Value'                                                            # otherwise medium

rfm['ValueCategory'] = rfm['Monetary'].apply(classify_value_tiers)                      # apply to create ValueCategory column

# ---------------------------
# 15. MAP INTO MARKETING SEGMENTS (6 actionable segments)
# ---------------------------
def determine_marketing_segment(row):                                                   # map combination of value + churn to a marketing segment
    if row['ValueCategory'] == 'High Value' and row['ChurnStatus'] == 'Active':         # high value and active
        return 'Champions'                                                               # VIPs / champions
    if row['ValueCategory'] == 'High Value' and row['ChurnStatus'] == 'Churned':        # high value but churned
        return 'At-Risk High Value'                                                      # reactivation targets (urgent)
    if row['ValueCategory'] == 'Medium Value' and row['ChurnStatus'] == 'Active':       # medium value active
        return 'Loyal Regulars'                                                          # upsell candidates
    if row['ValueCategory'] == 'Medium Value' and row['ChurnStatus'] == 'Churned':      # medium value churned
        return 'At-Risk Regulars'                                                        # win-back lists
    if row['ValueCategory'] == 'Low Value' and row['ChurnStatus'] == 'Active':          # low value but active
        return 'New / Occasional'                                                        # nurture list
    # fallback for Low Value & Churned or any other combination
    return 'Lost Customers'                                                              # lost or disengaged

rfm['MarketingSegment'] = rfm.apply(determine_marketing_segment, axis=1)               # create MarketingSegment column using mapping function

# ---------------------------
# 16. FLAG TOP and BOTTOM CUSTOMERS and RANK
# ---------------------------
rfm['Rank'] = rfm['Monetary'].rank(method='dense', ascending=False)                     # dense rank of monetary, 1 = largest spender
top_10 = rfm.nlargest(10, 'Monetary')                                                   # top 10 highest spenders
bottom_10 = rfm.nsmallest(10, 'Monetary')                                               # bottom 10 smallest spenders
rfm['CustomerFlag'] = np.where(rfm['CustomerID'].isin(top_10['CustomerID']), 'Top 10',  # flag top10
                        np.where(rfm['CustomerID'].isin(bottom_10['CustomerID']), 'Bottom 10', 'Normal'))  # flag bottom10 else normal

# ---------------------------
# 17. SEGMENT SUMMARY (business ready)
# ---------------------------
segment_summary = rfm.groupby('MarketingSegment').agg(                                  # aggregate metrics per marketing segment
    CustomerCount=('CustomerID', 'count'),                                             # number of customers in segment
    AvgRecency=('Recency', 'mean'),                                                     # average recency
    AvgFrequency=('Frequency', 'mean'),                                                 # average transaction frequency
    AvgMonetary=('Monetary', 'mean'),                                                   # average monetary per customer
    TotalMonetary=('Monetary', 'sum')                                                   # total revenue from segment
).sort_values(by='TotalMonetary', ascending=False).reset_index()                        # sort descending by contribution to revenue

print("\n===== MARKETING SEGMENT SUMMARY =====")                                         # header for console output
print(segment_summary)                                                                  # display segment summary

# ---------------------------
# 18. SAVE A RECENCY CHURN THRESHOLD VISUAL FOR MARKETING REVIEW
# ---------------------------
plt.figure(figsize=(8, 5))                                                              # prepare figure
sns.histplot(rfm['Recency'], bins=30, kde=True, color='skyblue', stat='count')         # show counts to reflect absolute customers on recency plot
plt.axvline(rec_churn_threshold, color='red', linestyle='--', label='Churn Threshold')  # mark churn threshold
plt.title('Recency Distribution with Automatic Churn Threshold')                       # title
plt.xlabel('Recency (days)')                                                            # x-axis label
plt.ylabel('Number of Customers')                                                       # y-axis label
plt.legend()                                                                            # show legend
plt.tight_layout()                                                                      # adjust layout
plt.savefig(os.path.join(OUTPUT_DIR, 'recency_churn_threshold.png'), dpi=200)           # save image in outputs folder
plt.close()                                                                             # close figure

# ---------------------------
# 19. EXPORT MASTER CSV + SEGMENT CSVs + CLUSTER CSV + TOP/BOTTOM CSVS
# ---------------------------
# ensure output directory exists (already created earlier but safe to call)
os.makedirs(OUTPUT_DIR, exist_ok=True)                                                  # ensure outputs folder exists

# Master file with all RFM, cluster, labels, and marketing segment columns
master_path = os.path.join(OUTPUT_DIR, 'RFM_Segmentation_Master_Full.csv')              # master csv path
rfm.to_csv(master_path, index=False)                                                    # export master file
print(f"✅ Master RFM file exported: {master_path}")                                     # log export success

# Export segment-specific CSVs for marketing use (one file per marketing segment)
segment_dir = os.path.join(OUTPUT_DIR, 'segments')                                      # subfolder for per-segment CSVs
os.makedirs(segment_dir, exist_ok=True)                                                 # create if missing

for seg_name, seg_df in rfm.groupby('MarketingSegment'):                               # iterate each marketing segment
    safe_seg_name = seg_name.replace('/', '_').replace(' ', '_')                       # sanitize file name
    seg_path = os.path.join(segment_dir, f"{safe_seg_name}.csv")                       # define path for this segment
    seg_df.to_csv(seg_path, index=False)                                                # write segment CSV
    print(f"📤 Exported segment: {seg_path}")                                           # log per-segment export

# Export segment summary table
summary_path = os.path.join(OUTPUT_DIR, 'RFM_Marketing_Segment_Summary.csv')           # summary csv path
segment_summary.to_csv(summary_path, index=False)                                      # write summary csv
print(f"📊 Exported segment summary: {summary_path}")                                   # log summary export

# Export clustering summary and profiles
cluster_summary_path = os.path.join(OUTPUT_DIR, 'cluster_profiles_full.csv')            # cluster profiles csv path (already saved earlier but re-export to OUTPUT_DIR)
cluster_summary.to_csv(cluster_summary_path, index=False)                              # write cluster profiles
print(f"📂 Exported cluster profile: {cluster_summary_path}")                           # log cluster export

# Export Top 10 and Bottom 10 lists for quick action
top_path = os.path.join(OUTPUT_DIR, 'Top_10_Customers.csv')                            # path for top 10
bottom_path = os.path.join(OUTPUT_DIR, 'Bottom_10_Customers.csv')                      # path for bottom 10
top_10.to_csv(top_path, index=False)                                                    # save top10 list
bottom_10.to_csv(bottom_path, index=False)                                              # save bottom10 list
print(f"🏆 Exported Top 10: {top_path}")                                                 # log
print(f"📉 Exported Bottom 10: {bottom_path}")                                           # log

# ---------------------------
# 20. OPTIONAL: EXPORT AN EXCEL WORKBOOK WITH SHEETS (one sheet per segment + master + summary)
# ---------------------------
excel_path = os.path.join(OUTPUT_DIR, 'RFM_Segmentation_Workbook.xlsx')                 # path for Excel workbook
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:                           # open an Excel writer context
    rfm.to_excel(writer, sheet_name='Master', index=False)                              # write master dataset to sheet 'Master'
    segment_summary.to_excel(writer, sheet_name='Summary', index=False)                 # write segment summary to 'Summary' sheet
    cluster_summary.to_excel(writer, sheet_name='ClusterProfiles', index=False)         # write cluster profiles to a sheet
    top_10.to_excel(writer, sheet_name='Top_10', index=False)                           # write top 10 to sheet
    bottom_10.to_excel(writer, sheet_name='Bottom_10', index=False)                     # write bottom 10 to sheet
    # write each marketing segment to its own sheet for easy review in Excel
    for seg_name, seg_df in rfm.groupby('MarketingSegment'):                            # iterate each marketing segment
        safe_seg_name = seg_name[:31].replace('/', '_').replace(' ', '_')               # sheet name must be <= 31 chars, sanitize
        seg_df.to_excel(writer, sheet_name=safe_seg_name, index=False)                  # write segment df to its sheet
print(f"📘 Excel workbook created: {excel_path}")                                       # log excel workbook creation

# ---------------------------
# 21. FINAL LOGS AND SUMMARY
# ---------------------------
print("\n===== FINAL OUTPUT SUMMARY =====")                                              # final header
print(f"Total customers analyzed: {len(rfm)}")                                          # total number of customers processed
print(f"Segments created: {rfm['MarketingSegment'].nunique()} -> {sorted(rfm['MarketingSegment'].unique())}")  # list segments
print(f"Master CSV: {master_path}")                                                      # master file path
print(f"Segments folder: {segment_dir}")                                                 # folder containing per-segment CSVs
print(f"Workbook: {excel_path}")                                                         # excel workbook path
print("✅ All exports complete. Review the files in the outputs folder for marketing and reporting.")  # completion message

# ===================================================================
# END OF COMBINED MASTER SCRIPT
# ===================================================================
