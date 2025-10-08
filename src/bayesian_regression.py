import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

# === Paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
csv_output_path = os.path.join(OUTPUT_DIR, 'customers_segmented_v2.csv')
bayesian_path = os.path.join(OUTPUT_DIR, 'bayesian_regression_v2.png')

# === Load clustered RFM data ===
if not os.path.exists(csv_output_path):
    raise FileNotFoundError(f"Missing file: {csv_output_path}")

rfm = pd.read_csv(csv_output_path)

# === Bayesian Linear Regression (Monetary ~ Frequency) ===
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    mu = alpha + beta * rfm['Frequency']
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=rfm['Monetary'])

    trace = pm.sample(1000, tune=1000, cores=1, progressbar=False)

# === Plot regression ===
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
