#!/usr/bin/env python3
"""
Phase 5 Expansion: Statistical Anomaly Detection Methods
Z-score, IQR, Mahalanobis distance, Chi-square
"""

import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

cells = []

# ============================================================================
# TITLE
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 📊 Statistical Anomaly Detection Methods\n",
        "\n",
        "**Phase 5 Expansion: Классические статистические подходы к обнаружению выбросов**\n",
        "\n",
        "---\n",
        "\n",
        "## 🎯 Зачем статистические методы?\n",
        "\n",
        "**Преимущества:**\n",
        "- ✅ Простые и интерпретируемые\n",
        "- ✅ Не требуют обучения\n",
        "- ✅ Быстрые (O(n))\n",
        "- ✅ Хорошо работают для univariate данных\n",
        "- ✅ Теоретически обоснованы\n",
        "\n",
        "**Когда использовать:**\n",
        "- Univariate или low-dimensional данные\n",
        "- Данные близки к нормальному распределению\n",
        "- Нужна интерпретируемость\n",
        "- Quick first pass перед ML методами\n",
        "\n",
        "---\n",
        "\n",
        "## 📋 Методы в этом notebook\n",
        "\n",
        "### 1. **Z-Score (Standard Score)**\n",
        "- Сколько стандартных отклонений от среднего\n",
        "- Threshold: обычно |z| > 3\n",
        "- Предположение: нормальное распределение\n",
        "\n",
        "### 2. **IQR (Interquartile Range)**\n",
        "- Robust к выбросам (использует медиану)\n",
        "- Threshold: < Q1 - 1.5*IQR или > Q3 + 1.5*IQR\n",
        "- Не требует нормальности\n",
        "\n",
        "### 3. **Modified Z-Score (MAD)**\n",
        "- Median Absolute Deviation вместо std\n",
        "- Robust версия Z-score\n",
        "\n",
        "### 4. **Mahalanobis Distance**\n",
        "- Multivariate outlier detection\n",
        "- Учитывает корреляции между признаками\n",
        "- Использует covariance matrix\n",
        "\n",
        "### 5. **Grubbs Test**\n",
        "- Statistical test для outliers\n",
        "- Hypothesis testing framework\n",
        "\n",
        "---\n"
    ]
})

# ============================================================================
# IMPORTS
# ============================================================================

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from scipy import stats\n",
        "from scipy.spatial.distance import mahalanobis\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.covariance import EmpiricalCovariance, MinCovDet\n",
        "from sklearn.datasets import make_blobs\n",
        "\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "np.random.seed(42)\n",
        "\n",
        "print(\"✅ Libraries loaded\")\n"
    ]
})

# ============================================================================
# DATASET
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 💾 Dataset: Manufacturing Quality Control\n",
        "\n",
        "Создадим реалистичный датасет измерений с производственной линии:\n",
        "- Нормальные измерения + выбросы (дефекты)\n",
        "- Несколько признаков с разными распределениями\n",
        "- Корреляции между признаками"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def create_manufacturing_data(n_samples=1000, anomaly_rate=0.05):\n",
        "    \"\"\"\n",
        "    Create manufacturing quality control dataset.\n",
        "    \n",
        "    Features:\n",
        "    - temperature: process temperature (correlated with pressure)\n",
        "    - pressure: process pressure\n",
        "    - vibration: machine vibration\n",
        "    - thickness: product thickness\n",
        "    - speed: production speed\n",
        "    \"\"\"\n",
        "    np.random.seed(42)\n",
        "    \n",
        "    n_normal = int(n_samples * (1 - anomaly_rate))\n",
        "    n_anomaly = n_samples - n_normal\n",
        "    \n",
        "    # Normal data - multivariate normal with correlations\n",
        "    mean = [100, 50, 0.5, 2.0, 100]\n",
        "    # Covariance matrix (temperature-pressure correlated)\n",
        "    cov = [\n",
        "        [25, 10, 0.1, 0.05, 5],    # temperature\n",
        "        [10, 16, 0.05, 0.02, 3],   # pressure\n",
        "        [0.1, 0.05, 0.01, 0.001, 0.02],  # vibration\n",
        "        [0.05, 0.02, 0.001, 0.04, 0.01], # thickness\n",
        "        [5, 3, 0.02, 0.01, 100]    # speed\n",
        "    ]\n",
        "    \n",
        "    normal_data = np.random.multivariate_normal(mean, cov, n_normal)\n",
        "    \n",
        "    # Anomalies - different types\n",
        "    anomalies = []\n",
        "    \n",
        "    # Type 1: High temperature anomaly\n",
        "    n1 = n_anomaly // 3\n",
        "    high_temp = np.random.multivariate_normal(\n",
        "        [130, 55, 0.6, 2.1, 95], \n",
        "        np.array(cov) * 0.5, \n",
        "        n1\n",
        "    )\n",
        "    anomalies.append(high_temp)\n",
        "    \n",
        "    # Type 2: High vibration anomaly\n",
        "    n2 = n_anomaly // 3\n",
        "    high_vib = np.random.multivariate_normal(\n",
        "        [102, 51, 1.2, 2.0, 100],\n",
        "        np.array(cov) * 0.3,\n",
        "        n2\n",
        "    )\n",
        "    anomalies.append(high_vib)\n",
        "    \n",
        "    # Type 3: Random outliers\n",
        "    n3 = n_anomaly - n1 - n2\n",
        "    random_outliers = np.random.multivariate_normal(\n",
        "        [80, 35, 0.3, 1.5, 130],\n",
        "        np.array(cov) * 2,\n",
        "        n3\n",
        "    )\n",
        "    anomalies.append(random_outliers)\n",
        "    \n",
        "    anomaly_data = np.vstack(anomalies)\n",
        "    \n",
        "    # Combine\n",
        "    X = np.vstack([normal_data, anomaly_data])\n",
        "    y = np.array([0] * n_normal + [1] * n_anomaly)  # 0=normal, 1=anomaly\n",
        "    \n",
        "    # Shuffle\n",
        "    idx = np.random.permutation(len(X))\n",
        "    X = X[idx]\n",
        "    y = y[idx]\n",
        "    \n",
        "    # Create DataFrame\n",
        "    df = pd.DataFrame(X, columns=['temperature', 'pressure', 'vibration', 'thickness', 'speed'])\n",
        "    df['is_anomaly'] = y\n",
        "    \n",
        "    return df\n",
        "\n",
        "# Create dataset\n",
        "df = create_manufacturing_data(n_samples=2000, anomaly_rate=0.05)\n",
        "\n",
        "print(\"📊 Manufacturing Quality Control Dataset\")\n",
        "print(\"=\" * 50)\n",
        "print(f\"Total samples: {len(df)}\")\n",
        "print(f\"Normal: {(df['is_anomaly'] == 0).sum()}\")\n",
        "print(f\"Anomalies: {(df['is_anomaly'] == 1).sum()} ({df['is_anomaly'].mean()*100:.1f}%)\")\n",
        "print(f\"\\nFeatures: {list(df.columns[:-1])}\")\n",
        "print(f\"\\nStatistics:\")\n",
        "print(df.describe().round(2))\n"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualize data\n",
        "fig, axes = plt.subplots(2, 3, figsize=(16, 10))\n",
        "\n",
        "feature_cols = ['temperature', 'pressure', 'vibration', 'thickness', 'speed']\n",
        "\n",
        "# Histograms\n",
        "for idx, col in enumerate(feature_cols):\n",
        "    ax = axes[idx // 3, idx % 3]\n",
        "    \n",
        "    # Normal\n",
        "    df[df['is_anomaly'] == 0][col].hist(\n",
        "        ax=ax, bins=30, alpha=0.6, label='Normal', color='blue'\n",
        "    )\n",
        "    # Anomaly\n",
        "    df[df['is_anomaly'] == 1][col].hist(\n",
        "        ax=ax, bins=30, alpha=0.6, label='Anomaly', color='red'\n",
        "    )\n",
        "    \n",
        "    ax.set_title(col.capitalize(), fontsize=12, fontweight='bold')\n",
        "    ax.set_xlabel(col)\n",
        "    ax.legend()\n",
        "\n",
        "# Correlation heatmap\n",
        "corr = df[feature_cols].corr()\n",
        "sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])\n",
        "axes[1, 2].set_title('Feature Correlations', fontsize=12, fontweight='bold')\n",
        "\n",
        "plt.suptitle('Manufacturing Data Distribution', fontsize=14, fontweight='bold', y=1.00)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n💡 Observations:\")\n",
        "print(\"- Temperature and Pressure are correlated\")\n",
        "print(\"- Anomalies visible in temperature and vibration distributions\")\n"
    ]
})

# ============================================================================
# Z-SCORE
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## 📏 Метод 1: Z-Score\n",
        "\n",
        "**Формула:**\n",
        "$$z = \\frac{x - \\mu}{\\sigma}$$\n",
        "\n",
        "**Интерпретация:**\n",
        "- z = 0: значение равно среднему\n",
        "- z = 1: на 1 std выше среднего\n",
        "- |z| > 3: выброс (99.7% данных в ±3σ)\n",
        "\n",
        "**Assumptions:**\n",
        "- Нормальное распределение\n",
        "- Независимые признаки\n",
        "\n",
        "---\n"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"=\" * 70)\n",
        "print(\"Z-SCORE ANOMALY DETECTION\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "def zscore_outliers(data, threshold=3):\n",
        "    \"\"\"\n",
        "    Detect outliers using Z-score.\n",
        "    \n",
        "    Args:\n",
        "        data: array-like\n",
        "        threshold: z-score threshold (default 3)\n",
        "    \n",
        "    Returns:\n",
        "        outlier_mask: boolean array\n",
        "        z_scores: z-scores for each point\n",
        "    \"\"\"\n",
        "    z_scores = np.abs(stats.zscore(data))\n",
        "    outlier_mask = z_scores > threshold\n",
        "    return outlier_mask, z_scores\n",
        "\n",
        "# Apply Z-score to each feature\n",
        "X = df[feature_cols].values\n",
        "y_true = df['is_anomaly'].values\n",
        "\n",
        "# Calculate Z-scores\n",
        "z_scores = np.abs(stats.zscore(X, axis=0))\n",
        "\n",
        "# Outlier if ANY feature has |z| > 3\n",
        "threshold = 3\n",
        "outlier_any = (z_scores > threshold).any(axis=1)\n",
        "\n",
        "# Outlier if MAX z-score > 3\n",
        "max_z = z_scores.max(axis=1)\n",
        "outlier_max = max_z > threshold\n",
        "\n",
        "print(f\"\\nThreshold: |z| > {threshold}\")\n",
        "print(f\"\\nOutliers detected (ANY feature): {outlier_any.sum()} ({outlier_any.mean()*100:.1f}%)\")\n",
        "print(f\"Outliers detected (MAX z-score): {outlier_max.sum()} ({outlier_max.mean()*100:.1f}%)\")\n",
        "\n",
        "# Evaluation\n",
        "from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix\n",
        "\n",
        "print(f\"\\n📊 Performance (using MAX z-score):\")\n",
        "print(f\"  Precision: {precision_score(y_true, outlier_max):.3f}\")\n",
        "print(f\"  Recall: {recall_score(y_true, outlier_max):.3f}\")\n",
        "print(f\"  F1-Score: {f1_score(y_true, outlier_max):.3f}\")\n",
        "\n",
        "print(f\"\\nConfusion Matrix:\")\n",
        "print(confusion_matrix(y_true, outlier_max))\n"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualize Z-scores\n",
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
        "\n",
        "# Z-score distribution\n",
        "axes[0, 0].hist(max_z, bins=50, edgecolor='black', alpha=0.7)\n",
        "axes[0, 0].axvline(x=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')\n",
        "axes[0, 0].set_title('Distribution of Max Z-Scores', fontsize=12, fontweight='bold')\n",
        "axes[0, 0].set_xlabel('Max Z-Score')\n",
        "axes[0, 0].set_ylabel('Count')\n",
        "axes[0, 0].legend()\n",
        "\n",
        "# Z-scores by feature\n",
        "z_df = pd.DataFrame(z_scores, columns=feature_cols)\n",
        "z_df.boxplot(ax=axes[0, 1])\n",
        "axes[0, 1].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')\n",
        "axes[0, 1].set_title('Z-Scores by Feature', fontsize=12, fontweight='bold')\n",
        "axes[0, 1].set_ylabel('|Z-Score|')\n",
        "axes[0, 1].legend()\n",
        "\n",
        "# Scatter: Temperature vs Pressure\n",
        "colors = np.where(outlier_max, 'red', 'blue')\n",
        "axes[1, 0].scatter(df['temperature'], df['pressure'], c=colors, alpha=0.5, s=20)\n",
        "axes[1, 0].set_title('Z-Score Detection: Temp vs Pressure', fontsize=12, fontweight='bold')\n",
        "axes[1, 0].set_xlabel('Temperature')\n",
        "axes[1, 0].set_ylabel('Pressure')\n",
        "\n",
        "# ROC-like: threshold sensitivity\n",
        "thresholds = np.linspace(1, 5, 50)\n",
        "precisions = []\n",
        "recalls = []\n",
        "\n",
        "for t in thresholds:\n",
        "    pred = max_z > t\n",
        "    if pred.sum() > 0:\n",
        "        precisions.append(precision_score(y_true, pred))\n",
        "        recalls.append(recall_score(y_true, pred))\n",
        "    else:\n",
        "        precisions.append(0)\n",
        "        recalls.append(0)\n",
        "\n",
        "axes[1, 1].plot(thresholds, precisions, label='Precision', linewidth=2)\n",
        "axes[1, 1].plot(thresholds, recalls, label='Recall', linewidth=2)\n",
        "axes[1, 1].axvline(x=3, color='gray', linestyle='--', alpha=0.5)\n",
        "axes[1, 1].set_title('Precision/Recall vs Threshold', fontsize=12, fontweight='bold')\n",
        "axes[1, 1].set_xlabel('Z-Score Threshold')\n",
        "axes[1, 1].set_ylabel('Score')\n",
        "axes[1, 1].legend()\n",
        "axes[1, 1].grid(alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
})

# ============================================================================
# IQR METHOD
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## 📦 Метод 2: IQR (Interquartile Range)\n",
        "\n",
        "**Формула:**\n",
        "- Q1 = 25th percentile\n",
        "- Q3 = 75th percentile\n",
        "- IQR = Q3 - Q1\n",
        "- Lower bound = Q1 - 1.5 * IQR\n",
        "- Upper bound = Q3 + 1.5 * IQR\n",
        "\n",
        "**Преимущества:**\n",
        "- Robust к выбросам (использует медиану)\n",
        "- Не требует нормальности\n",
        "- Основа для boxplot\n",
        "\n",
        "---\n"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"=\" * 70)\n",
        "print(\"IQR (INTERQUARTILE RANGE) METHOD\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "def iqr_outliers(data, k=1.5):\n",
        "    \"\"\"\n",
        "    Detect outliers using IQR method.\n",
        "    \n",
        "    Args:\n",
        "        data: array-like (1D or 2D)\n",
        "        k: multiplier for IQR (default 1.5, use 3 for extreme)\n",
        "    \n",
        "    Returns:\n",
        "        outlier_mask: boolean array\n",
        "    \"\"\"\n",
        "    if data.ndim == 1:\n",
        "        data = data.reshape(-1, 1)\n",
        "    \n",
        "    Q1 = np.percentile(data, 25, axis=0)\n",
        "    Q3 = np.percentile(data, 75, axis=0)\n",
        "    IQR = Q3 - Q1\n",
        "    \n",
        "    lower = Q1 - k * IQR\n",
        "    upper = Q3 + k * IQR\n",
        "    \n",
        "    outlier_mask = ((data < lower) | (data > upper)).any(axis=1)\n",
        "    \n",
        "    return outlier_mask, lower, upper\n",
        "\n",
        "# Apply IQR\n",
        "iqr_outliers_mask, lower_bounds, upper_bounds = iqr_outliers(X, k=1.5)\n",
        "\n",
        "print(f\"\\nk = 1.5 (standard)\")\n",
        "print(f\"Outliers detected: {iqr_outliers_mask.sum()} ({iqr_outliers_mask.mean()*100:.1f}%)\")\n",
        "\n",
        "# Evaluation\n",
        "print(f\"\\n📊 Performance:\")\n",
        "print(f\"  Precision: {precision_score(y_true, iqr_outliers_mask):.3f}\")\n",
        "print(f\"  Recall: {recall_score(y_true, iqr_outliers_mask):.3f}\")\n",
        "print(f\"  F1-Score: {f1_score(y_true, iqr_outliers_mask):.3f}\")\n",
        "\n",
        "# Show bounds for each feature\n",
        "print(f\"\\nBounds for each feature:\")\n",
        "for i, col in enumerate(feature_cols):\n",
        "    print(f\"  {col}: [{lower_bounds[i]:.2f}, {upper_bounds[i]:.2f}]\")\n"
    ]
})

# ============================================================================
# MAHALANOBIS DISTANCE
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## 📐 Метод 3: Mahalanobis Distance\n",
        "\n",
        "**Multivariate outlier detection!**\n",
        "\n",
        "**Формула:**\n",
        "$$D_M = \\sqrt{(x - \\mu)^T \\Sigma^{-1} (x - \\mu)}$$\n",
        "\n",
        "**Преимущества:**\n",
        "- Учитывает корреляции между признаками\n",
        "- Scale-invariant\n",
        "- Находит multivariate outliers\n",
        "\n",
        "**Интерпретация:**\n",
        "- D² ~ χ²(p) где p = число признаков\n",
        "- Threshold: χ²(p, 0.975) для 97.5% confidence\n",
        "\n",
        "---\n"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"=\" * 70)\n",
        "print(\"MAHALANOBIS DISTANCE\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "def mahalanobis_outliers(X, contamination=0.05):\n",
        "    \"\"\"\n",
        "    Detect outliers using Mahalanobis distance.\n",
        "    \n",
        "    Uses Minimum Covariance Determinant for robust estimation.\n",
        "    \"\"\"\n",
        "    # Robust covariance estimation\n",
        "    robust_cov = MinCovDet(random_state=42).fit(X)\n",
        "    \n",
        "    # Mahalanobis distances\n",
        "    mahal_dist = robust_cov.mahalanobis(X)\n",
        "    \n",
        "    # Threshold using chi-squared distribution\n",
        "    # D² follows chi-squared with p degrees of freedom\n",
        "    p = X.shape[1]\n",
        "    threshold = stats.chi2.ppf(1 - contamination, p)\n",
        "    \n",
        "    outlier_mask = mahal_dist > threshold\n",
        "    \n",
        "    return outlier_mask, mahal_dist, threshold\n",
        "\n",
        "# Apply Mahalanobis\n",
        "mahal_outliers, mahal_dist, mahal_threshold = mahalanobis_outliers(X, contamination=0.05)\n",
        "\n",
        "print(f\"\\nThreshold (χ²): {mahal_threshold:.2f}\")\n",
        "print(f\"Outliers detected: {mahal_outliers.sum()} ({mahal_outliers.mean()*100:.1f}%)\")\n",
        "\n",
        "# Evaluation\n",
        "print(f\"\\n📊 Performance:\")\n",
        "print(f\"  Precision: {precision_score(y_true, mahal_outliers):.3f}\")\n",
        "print(f\"  Recall: {recall_score(y_true, mahal_outliers):.3f}\")\n",
        "print(f\"  F1-Score: {f1_score(y_true, mahal_outliers):.3f}\")\n",
        "\n",
        "print(f\"\\nConfusion Matrix:\")\n",
        "print(confusion_matrix(y_true, mahal_outliers))\n"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualize Mahalanobis\n",
        "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n",
        "\n",
        "# Distribution of distances\n",
        "axes[0].hist(mahal_dist, bins=50, edgecolor='black', alpha=0.7)\n",
        "axes[0].axvline(x=mahal_threshold, color='r', linestyle='--', \n",
        "                label=f'Threshold={mahal_threshold:.1f}')\n",
        "axes[0].set_title('Mahalanobis Distance Distribution', fontsize=12, fontweight='bold')\n",
        "axes[0].set_xlabel('Mahalanobis Distance')\n",
        "axes[0].set_ylabel('Count')\n",
        "axes[0].legend()\n",
        "\n",
        "# Scatter with Mahalanobis coloring\n",
        "scatter = axes[1].scatter(df['temperature'], df['pressure'], \n",
        "                         c=mahal_dist, cmap='viridis', alpha=0.6, s=20)\n",
        "plt.colorbar(scatter, ax=axes[1], label='Mahal. Distance')\n",
        "axes[1].set_title('Mahalanobis Distance: Temp vs Pressure', fontsize=12, fontweight='bold')\n",
        "axes[1].set_xlabel('Temperature')\n",
        "axes[1].set_ylabel('Pressure')\n",
        "\n",
        "# Detected outliers\n",
        "colors = np.where(mahal_outliers, 'red', 'blue')\n",
        "axes[2].scatter(df['temperature'], df['vibration'], c=colors, alpha=0.5, s=20)\n",
        "axes[2].set_title('Mahalanobis Detection: Temp vs Vibration', fontsize=12, fontweight='bold')\n",
        "axes[2].set_xlabel('Temperature')\n",
        "axes[2].set_ylabel('Vibration')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n💡 Mahalanobis advantage:\")\n",
        "print(\"- Detects multivariate outliers that univariate methods miss\")\n",
        "print(\"- Accounts for correlation between temperature and pressure\")\n"
    ]
})

# ============================================================================
# COMPARISON
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## 📊 Comparison of All Methods"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"=\" * 70)\n",
        "print(\"METHOD COMPARISON\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "# All methods\n",
        "methods = {\n",
        "    'Z-Score (|z|>3)': outlier_max,\n",
        "    'IQR (k=1.5)': iqr_outliers_mask,\n",
        "    'Mahalanobis': mahal_outliers\n",
        "}\n",
        "\n",
        "# Results table\n",
        "results = []\n",
        "for name, pred in methods.items():\n",
        "    results.append({\n",
        "        'Method': name,\n",
        "        'Detected': pred.sum(),\n",
        "        'Precision': precision_score(y_true, pred),\n",
        "        'Recall': recall_score(y_true, pred),\n",
        "        'F1': f1_score(y_true, pred)\n",
        "    })\n",
        "\n",
        "results_df = pd.DataFrame(results)\n",
        "print(\"\\n\" + results_df.to_string(index=False))\n",
        "\n",
        "# Visualization\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# Metrics comparison\n",
        "x = np.arange(len(methods))\n",
        "width = 0.25\n",
        "\n",
        "axes[0].bar(x - width, results_df['Precision'], width, label='Precision', alpha=0.8)\n",
        "axes[0].bar(x, results_df['Recall'], width, label='Recall', alpha=0.8)\n",
        "axes[0].bar(x + width, results_df['F1'], width, label='F1', alpha=0.8)\n",
        "\n",
        "axes[0].set_xticks(x)\n",
        "axes[0].set_xticklabels(results_df['Method'], rotation=15)\n",
        "axes[0].set_ylabel('Score')\n",
        "axes[0].set_title('Method Comparison', fontsize=14, fontweight='bold')\n",
        "axes[0].legend()\n",
        "axes[0].grid(axis='y', alpha=0.3)\n",
        "\n",
        "# Venn-like: overlap of detections\n",
        "from matplotlib_venn import venn3\n",
        "try:\n",
        "    z_set = set(np.where(outlier_max)[0])\n",
        "    iqr_set = set(np.where(iqr_outliers_mask)[0])\n",
        "    mahal_set = set(np.where(mahal_outliers)[0])\n",
        "    \n",
        "    venn3([z_set, iqr_set, mahal_set], \n",
        "          set_labels=('Z-Score', 'IQR', 'Mahalanobis'),\n",
        "          ax=axes[1])\n",
        "    axes[1].set_title('Overlap of Detected Outliers', fontsize=14, fontweight='bold')\n",
        "except:\n",
        "    # Fallback if venn not available\n",
        "    axes[1].text(0.5, 0.5, 'Install matplotlib-venn\\nfor Venn diagram', \n",
        "                ha='center', va='center', fontsize=12)\n",
        "    axes[1].axis('off')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n💡 Conclusions:\")\n",
        "print(\"- Mahalanobis: Best for multivariate data with correlations\")\n",
        "print(\"- Z-Score: Simple, good for normally distributed features\")\n",
        "print(\"- IQR: Robust, good for skewed distributions\")\n"
    ]
})

# ============================================================================
# CONCLUSIONS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## 📋 Summary & Recommendations\n",
        "\n",
        "### When to use each method:\n",
        "\n",
        "| Method | Best for | Assumptions | Complexity |\n",
        "|--------|----------|-------------|------------|\n",
        "| **Z-Score** | Normal data, quick check | Normality | O(n) |\n",
        "| **IQR** | Skewed data, robust | None | O(n) |\n",
        "| **Modified Z (MAD)** | Robust Z-score | None | O(n) |\n",
        "| **Mahalanobis** | Multivariate, correlated | Multivariate normal | O(n·p²) |\n",
        "| **Grubbs Test** | Single outlier testing | Normality | O(n) |\n",
        "\n",
        "### Practical workflow:\n",
        "\n",
        "1. **Start with univariate:** Z-score or IQR per feature\n",
        "2. **Check multivariate:** Mahalanobis for correlated features\n",
        "3. **Combine:** Union or intersection of methods\n",
        "4. **Validate:** Check against domain knowledge\n",
        "5. **Iterate:** Adjust thresholds based on results\n",
        "\n",
        "### Key insights:\n",
        "\n",
        "- ✅ Statistical methods are fast and interpretable\n",
        "- ✅ Use Mahalanobis for multivariate data\n",
        "- ✅ IQR is more robust than Z-score\n",
        "- ⚠️ May miss complex non-linear anomalies\n",
        "- ⚠️ Sensitive to threshold selection\n",
        "\n",
        "**Next:** Combine with ML methods (Isolation Forest, Autoencoders) for better results!\n",
        "\n",
        "---\n"
    ]
})

# Save notebook
notebook['cells'] = cells

output_path = '/home/user/test/notebooks/phase5_anomaly_patterns/03_statistical_anomalies.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Statistical Anomalies notebook created: {output_path}')
print(f'Total cells: {len(cells)}')
