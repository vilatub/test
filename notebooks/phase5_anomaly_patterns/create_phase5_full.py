#!/usr/bin/env python3
"""
Phase 5: Complete Anomaly Detection & Clustering Notebook
Full implementation with synthetic credit card fraud data
"""

import json

# Load existing notebook
notebook_path = '/home/user/test/notebooks/phase5_anomaly_patterns/01_anomaly_detection_clustering.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

cells = notebook['cells']

# ============================================================================
# SYNTHETIC DATASET CREATION
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.2 Создание Synthetic Credit Card Dataset\n",
        "\n",
        "**Realistic fraud detection scenario:**\n",
        "- 50,000 transactions\n",
        "- ~0.2% fraud rate (100 fraud из 50,000)\n",
        "- PCA-transformed features (V1-V10)\n",
        "- Amount, Time features\n",
        "- Fraud patterns: unusual amounts, unusual times, specific feature combinations"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def create_fraud_dataset(n_samples=50000, fraud_rate=0.002):\n",
        "    \"\"\"\n",
        "    Create synthetic credit card fraud dataset\n",
        "    Similar to Kaggle Credit Card Fraud dataset structure\n",
        "    \"\"\"\n",
        "    np.random.seed(42)\n",
        "    \n",
        "    n_fraud = int(n_samples * fraud_rate)\n",
        "    n_normal = n_samples - n_fraud\n",
        "    \n",
        "    print(f\"Creating dataset: {n_samples:,} transactions\")\n",
        "    print(f\"  Normal: {n_normal:,} ({n_normal/n_samples*100:.2f}%)\")\n",
        "    print(f\"  Fraud:  {n_fraud:,} ({n_fraud/n_samples*100:.2f}%)\")\n",
        "    \n",
        "    # Normal transactions\n",
        "    normal_data = np.random.randn(n_normal, 10) * 2  # 10 PCA features\n",
        "    normal_time = np.random.uniform(0, 172800, n_normal)  # 48 hours in seconds\n",
        "    normal_amount = np.random.lognormal(3, 1.5, n_normal)  # log-normal distribution\n",
        "    normal_amount = np.clip(normal_amount, 1, 1000)\n",
        "    \n",
        "    # Fraud transactions (different patterns)\n",
        "    fraud_data = np.random.randn(n_fraud, 10) * 3 + 5  # shifted mean, higher variance\n",
        "    # Some fraud features more extreme\n",
        "    fraud_data[:, 0] += np.random.uniform(3, 8, n_fraud)  # V1 extreme\n",
        "    fraud_data[:, 3] += np.random.uniform(-8, -3, n_fraud)  # V4 extreme\n",
        "    \n",
        "    fraud_time = np.random.uniform(0, 172800, n_fraud)\n",
        "    # Fraud tends to have unusual amounts\n",
        "    fraud_amount = np.concatenate([\n",
        "        np.random.uniform(0.5, 5, n_fraud//2),  # very small\n",
        "        np.random.uniform(500, 5000, n_fraud//2)  # very large\n",
        "    ])\n",
        "    np.random.shuffle(fraud_amount)\n",
        "    \n",
        "    # Combine\n",
        "    X = np.vstack([normal_data, fraud_data])\n",
        "    time = np.concatenate([normal_time, fraud_time])\n",
        "    amount = np.concatenate([normal_amount, fraud_amount])\n",
        "    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])\n",
        "    \n",
        "    # Create DataFrame\n",
        "    df = pd.DataFrame(X, columns=[f'V{i+1}' for i in range(10)])\n",
        "    df['Time'] = time\n",
        "    df['Amount'] = amount\n",
        "    df['Class'] = y.astype(int)\n",
        "    \n",
        "    # Shuffle\n",
        "    df = df.sample(frac=1, random_state=42).reset_index(drop=True)\n",
        "    \n",
        "    print(f\"\\n✅ Dataset created: {len(df):,} rows, {len(df.columns)} columns\")\n",
        "    return df\n",
        "\n",
        "# Create dataset\n",
        "df = create_fraud_dataset(n_samples=50000, fraud_rate=0.002)\n",
        "\n",
        "print(f\"\\nDataset shape: {df.shape}\")\n",
        "print(f\"\\nFirst few rows:\")\n",
        "df.head()"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.3 Exploratory Data Analysis"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Basic statistics\n",
        "print(\"Dataset Statistics:\")\n",
        "print(\"=\"*60)\n",
        "print(f\"Total transactions: {len(df):,}\")\n",
        "print(f\"\\nClass distribution:\")\n",
        "print(df['Class'].value_counts())\n",
        "print(f\"\\nFraud rate: {df['Class'].mean()*100:.3f}%\")\n",
        "print(\"\\nAmount statistics:\")\n",
        "print(df.groupby('Class')['Amount'].describe())\n",
        "print(\"=\"*60)"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualizations\n",
        "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n",
        "\n",
        "# 1. Class distribution\n",
        "df['Class'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['steelblue', 'red'])\n",
        "axes[0, 0].set_title('Class Distribution (Extreme Imbalance!)', fontsize=14, fontweight='bold')\n",
        "axes[0, 0].set_xlabel('Class (0=Normal, 1=Fraud)')\n",
        "axes[0, 0].set_ylabel('Count')\n",
        "axes[0, 0].set_xticklabels(['Normal', 'Fraud'], rotation=0)\n",
        "\n",
        "# 2. Amount distribution\n",
        "axes[0, 1].hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.6, label='Normal', color='steelblue')\n",
        "axes[0, 1].hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.6, label='Fraud', color='red')\n",
        "axes[0, 1].set_title('Amount Distribution', fontsize=14, fontweight='bold')\n",
        "axes[0, 1].set_xlabel('Amount')\n",
        "axes[0, 1].set_ylabel('Frequency')\n",
        "axes[0, 1].legend()\n",
        "axes[0, 1].set_yscale('log')\n",
        "\n",
        "# 3. Time distribution\n",
        "axes[1, 0].hist(df[df['Class']==0]['Time'], bins=50, alpha=0.6, label='Normal', color='steelblue')\n",
        "axes[1, 0].hist(df[df['Class']==1]['Time'], bins=50, alpha=0.6, label='Fraud', color='red')\n",
        "axes[1, 0].set_title('Time Distribution', fontsize=14, fontweight='bold')\n",
        "axes[1, 0].set_xlabel('Time (seconds)')\n",
        "axes[1, 0].legend()\n",
        "\n",
        "# 4. V1 vs V4 (PCA features)\n",
        "normal = df[df['Class']==0].sample(1000, random_state=42)\n",
        "fraud = df[df['Class']==1]\n",
        "axes[1, 1].scatter(normal['V1'], normal['V4'], alpha=0.3, s=10, label='Normal', color='steelblue')\n",
        "axes[1, 1].scatter(fraud['V1'], fraud['V4'], alpha=0.7, s=30, label='Fraud', color='red', edgecolors='black')\n",
        "axes[1, 1].set_title('V1 vs V4 (sample)', fontsize=14, fontweight='bold')\n",
        "axes[1, 1].set_xlabel('V1')\n",
        "axes[1, 1].set_ylabel('V4')\n",
        "axes[1, 1].legend()\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n📊 Key Observations:\")\n",
        "print(\"  - Extreme class imbalance (~0.2% fraud) - realistic!\")\n",
        "print(\"  - Fraud has different amount distribution (very small or very large)\")\n",
        "print(\"  - Fraud clearly separable in some feature dimensions (V1, V4)\")\n",
        "print(\"  - Perfect use case for anomaly detection!\")"
    ]
})

# Continue adding more cells...
# Due to space, I'll add key sections in sequence

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## 🔍 Часть 2: Clustering Methods\n",
        "\n",
        "### 2.1 Data Preparation"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Prepare features for clustering\n",
        "# Use subset for computational efficiency\n",
        "sample_size = 10000\n",
        "df_sample = df.sample(sample_size, random_state=42).copy()\n",
        "\n",
        "feature_cols = [f'V{i+1}' for i in range(10)] + ['Time', 'Amount']\n",
        "X = df_sample[feature_cols].values\n",
        "y_true = df_sample['Class'].values\n",
        "\n",
        "# Standardize\n",
        "scaler = StandardScaler()\n",
        "X_scaled = scaler.fit_transform(X)\n",
        "\n",
        "print(f\"Sample for clustering: {X_scaled.shape}\")\n",
        "print(f\"Features: {feature_cols}\")\n",
        "print(f\"True fraud in sample: {y_true.sum()} ({y_true.mean()*100:.2f}%)\")"
    ]
})

# Save progress
notebook['cells'] = cells
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Progress saved: {len(cells)} cells')
print('Continuing with full implementation...')
