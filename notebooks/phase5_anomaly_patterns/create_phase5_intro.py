#!/usr/bin/env python3
"""
Phase 5: Anomaly Detection, Clustering & Pattern Mining
Part 1: Introduction, Dataset, Clustering Methods
"""

import json

# Создаем базовую структуру ноутбука
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
# TITLE AND INTRODUCTION
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🔍 Anomaly Detection, Clustering & Pattern Mining\n",
        "\n",
        "**Phase 5: Unsupervised Learning для Real-World Problems**\n",
        "\n",
        "---\n",
        "\n",
        "## 🎯 Проблемы Real World\n",
        "\n",
        "### До сих пор мы решали supervised tasks:\n",
        "\n",
        "- ✅ **Classification**: есть labels (survived/not survived, >50K/<=50K)\n",
        "- ✅ **Regression**: есть target (electricity consumption, price)\n",
        "- ✅ **Time Series Forecasting**: предсказываем future values\n",
        "\n",
        "**Но в реальном мире:**\n",
        "\n",
        "- ❓ **Fraud Detection**: 99.9% transactions нормальные, 0.1% - fraud (крайний дисбаланс!)\n",
        "- ❓ **Customer Segmentation**: нет заранее определённых групп\n",
        "- ❓ **Network Intrusions**: большинство событий нормальные, аномалии редки\n",
        "- ❓ **Equipment Failures**: failure events редки, но критичны\n",
        "- ❓ **New Attack Types**: не видели раньше, нет labels\n",
        "\n",
        "**Проблемы:**\n",
        "1. **No labels** или очень мало labeled data\n",
        "2. **Class imbalance**: anomalies составляют 0.001% - 1%\n",
        "3. **Novel patterns**: новые типы аномалий, которых не было в training\n",
        "4. **Interpretability**: нужно объяснить, ПОЧЕМУ что-то аномалия\n",
        "\n",
        "---\n",
        "\n",
        "## 🚀 Enter Unsupervised Learning\n",
        "\n",
        "### 1. Clustering (Кластеризация)\n",
        "\n",
        "**Задача:** Группировка похожих объектов без labels\n",
        "\n",
        "**Use Cases:**\n",
        "- 🛒 **Customer Segmentation**: группы клиентов с похожим поведением\n",
        "- 🏥 **Patient Stratification**: группы пациентов для personalized treatment\n",
        "- 📄 **Document Clustering**: темы в коллекции текстов\n",
        "- 🎵 **Music Recommendation**: похожие песни/исполнители\n",
        "\n",
        "**Методы:**\n",
        "- **K-Means**: partition-based, fast, assumes spherical clusters\n",
        "- **DBSCAN**: density-based, finds arbitrary shapes, handles outliers\n",
        "- **Hierarchical**: creates dendrogram, no need to specify K\n",
        "\n",
        "---\n",
        "\n",
        "### 2. Anomaly Detection (Обнаружение аномалий)\n",
        "\n",
        "**Задача:** Найти редкие, необычные observations\n",
        "\n",
        "**Use Cases:**\n",
        "- 💳 **Fraud Detection**: необычные транзакции\n",
        "- 🏭 **Predictive Maintenance**: аномальные sensor readings → failure prediction\n",
        "- 🔒 **Cybersecurity**: intrusion detection, DDoS attacks\n",
        "- 🏥 **Healthcare**: rare diseases, abnormal vitals\n",
        "- 📊 **Finance**: market manipulation, insider trading\n",
        "\n",
        "**Методы:**\n",
        "- **Isolation Forest**: isolate anomalies через random partitioning\n",
        "- **LOF (Local Outlier Factor)**: density-based, локальные outliers\n",
        "- **One-Class SVM**: learn boundary of \"normal\" data\n",
        "- **Autoencoders**: reconstruction error для аномалий\n",
        "- **Statistical**: Z-score, IQR, Mahalanobis distance\n",
        "\n",
        "---\n",
        "\n",
        "### 3. Pattern Mining\n",
        "\n",
        "**Задача:** Найти часто встречающиеся комбинации\n",
        "\n",
        "**Use Cases:**\n",
        "- 🛒 **Market Basket Analysis**: \"люди, купившие X, также покупают Y\"\n",
        "- 📊 **Feature Engineering**: automatic feature interactions\n",
        "- 🔗 **Recommendation**: association rules\n",
        "\n",
        "**Методы:**\n",
        "- **Apriori**: frequent itemsets\n",
        "- **FP-Growth**: faster alternative\n",
        "\n",
        "---\n",
        "\n",
        "## 📊 Что мы реализуем\n",
        "\n",
        "### Dataset: Credit Card Transactions (Synthetic)\n",
        "\n",
        "**Почему credit card fraud?**\n",
        "- ✅ Real-world problem (миллиарды убытков ежегодно)\n",
        "- ✅ Extreme class imbalance (~0.1% fraud)\n",
        "- ✅ Unlabeled data в production (новые fraud patterns)\n",
        "- ✅ Нужна interpretability (объяснить клиенту)\n",
        "\n",
        "**Создадим синтетические данные:**\n",
        "- ~50,000 transactions\n",
        "- ~0.2% fraud (realistic ratio)\n",
        "- Features: amount, time, merchant category, location, etc.\n",
        "- PCA-transformed features (как в real Kaggle dataset)\n",
        "\n",
        "**Задачи:**\n",
        "1. **Clustering**: Группировка transactions (normal spending patterns)\n",
        "2. **Anomaly Detection**: Найти fraud без labels\n",
        "3. **Comparison**: Multiple methods (Isolation Forest, LOF, etc.)\n",
        "4. **Visualization**: t-SNE/UMAP для interpretability\n",
        "\n",
        "---\n"
    ]
})

# ============================================================================
# IMPORTS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 💻 Часть 1: Setup и Dataset\n",
        "\n",
        "### 1.1 Импорт библиотек"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Базовые библиотеки\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Sklearn - Clustering\n",
        "from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering\n",
        "from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score\n",
        "\n",
        "# Sklearn - Anomaly Detection\n",
        "from sklearn.ensemble import IsolationForest\n",
        "from sklearn.neighbors import LocalOutlierFactor\n",
        "from sklearn.svm import OneClassSVM\n",
        "from sklearn.covariance import EllipticEnvelope\n",
        "\n",
        "# Sklearn - Preprocessing & Metrics\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.decomposition import PCA\n",
        "from sklearn.manifold import TSNE\n",
        "from sklearn.metrics import (\n",
        "    precision_score, recall_score, f1_score, \n",
        "    roc_auc_score, average_precision_score,\n",
        "    confusion_matrix, classification_report\n",
        ")\n",
        "\n",
        "# Scipy\n",
        "from scipy.cluster.hierarchy import dendrogram, linkage\n",
        "from scipy.stats import zscore\n",
        "\n",
        "# PyTorch (для Autoencoders)\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import Dataset, DataLoader, TensorDataset\n",
        "\n",
        "# Настройки\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "sns.set_palette(\"husl\")\n",
        "%matplotlib inline\n",
        "\n",
        "# Device\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f\"Device: {device}\")\n",
        "\n",
        "# Reproducibility\n",
        "np.random.seed(42)\n",
        "torch.manual_seed(42)\n",
        "\n",
        "print(\"\\n✅ Все библиотеки загружены\")"
    ]
})

# Сохраняем и продолжу в следующем файле из-за размера
notebook['cells'] = cells

output_path = '/home/user/test/notebooks/phase5_anomaly_patterns/01_anomaly_detection_clustering.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Started notebook: {output_path}')
print(f'Ячеек: {len(cells)}')
print('Продолжаю создание...')
