#!/usr/bin/env python3
"""
Add clustering and anomaly detection methods to Phase 5 notebook
"""

import json

notebook_path = '/home/user/test/notebooks/phase5_anomaly_patterns/01_anomaly_detection_clustering.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

cells = notebook['cells']

# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.2 K-Means Clustering"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# K-Means with different k values\n",
        "k_values = [2, 3, 5, 8]\n",
        "kmeans_results = {}\n",
        "\n",
        "for k in k_values:\n",
        "    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
        "    labels = kmeans.fit_predict(X_scaled)\n",
        "    \n",
        "    # Metrics\n",
        "    silhouette = silhouette_score(X_scaled, labels)\n",
        "    calinski = calinski_harabasz_score(X_scaled, labels)\n",
        "    davies = davies_bouldin_score(X_scaled, labels)\n",
        "    \n",
        "    kmeans_results[k] = {\n",
        "        'labels': labels,\n",
        "        'silhouette': silhouette,\n",
        "        'calinski': calinski,\n",
        "        'davies': davies,\n",
        "        'centers': kmeans.cluster_centers_\n",
        "    }\n",
        "    \n",
        "    print(f\"K={k}: Silhouette={silhouette:.3f}, Calinski-Harabasz={calinski:.1f}, Davies-Bouldin={davies:.3f}\")\n",
        "\n",
        "print(\"\\n📊 Best K by Silhouette Score (higher is better):\")\n",
        "best_k = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['silhouette'])\n",
        "print(f\"  K={best_k} with Silhouette={kmeans_results[best_k]['silhouette']:.3f}\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualize K-Means results (K=3)\n",
        "k = 3\n",
        "labels_km = kmeans_results[k]['labels']\n",
        "\n",
        "# PCA для visualization\n",
        "pca_vis = PCA(n_components=2, random_state=42)\n",
        "X_pca = pca_vis.fit_transform(X_scaled)\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "\n",
        "# K-Means clusters\n",
        "for cluster_id in range(k):\n",
        "    mask = labels_km == cluster_id\n",
        "    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], \n",
        "                   alpha=0.5, s=20, label=f'Cluster {cluster_id}')\n",
        "axes[0].set_title(f'K-Means Clustering (K={k})', fontsize=14, fontweight='bold')\n",
        "axes[0].set_xlabel('PC1')\n",
        "axes[0].set_ylabel('PC2')\n",
        "axes[0].legend()\n",
        "\n",
        "# True labels (fraud vs normal)\n",
        "axes[1].scatter(X_pca[y_true==0, 0], X_pca[y_true==0, 1], \n",
        "               alpha=0.3, s=10, label='Normal', color='steelblue')\n",
        "axes[1].scatter(X_pca[y_true==1, 0], X_pca[y_true==1, 1], \n",
        "               alpha=0.8, s=50, label='Fraud', color='red', edgecolors='black')\n",
        "axes[1].set_title('True Labels (Normal vs Fraud)', fontsize=14, fontweight='bold')\n",
        "axes[1].set_xlabel('PC1')\n",
        "axes[1].set_ylabel('PC2')\n",
        "axes[1].legend()\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Fraud distribution across clusters\n",
        "print(\"\\nFraud distribution across K-Means clusters:\")\n",
        "for cluster_id in range(k):\n",
        "    mask = labels_km == cluster_id\n",
        "    n_total = mask.sum()\n",
        "    n_fraud = y_true[mask].sum()\n",
        "    fraud_rate = n_fraud / n_total if n_total > 0 else 0\n",
        "    print(f\"  Cluster {cluster_id}: {n_fraud}/{n_total} fraud ({fraud_rate*100:.2f}%)\")"
    ]
})

# ============================================================================
# ISOLATION FOREST
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## 🚨 Часть 3: Anomaly Detection Methods\n",
        "\n",
        "### 3.1 Isolation Forest\n",
        "\n",
        "**Принцип:** Anomalies легче изолировать (меньше splits нужно)\n",
        "\n",
        "**Как работает:**\n",
        "1. Случайно выбирает feature\n",
        "2. Случайно выбирает split value\n",
        "3. Аномалии быстро изолируются (короткий path)\n",
        "4. Normal points требуют больше splits\n",
        "\n",
        "**Anomaly Score:** Среднее path length (меньше = более аномально)"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Isolation Forest\n",
        "# contamination = expected fraud rate\n",
        "iso_forest = IsolationForest(\n",
        "    contamination=0.002,  # expected fraud rate\n",
        "    random_state=42,\n",
        "    n_estimators=100\n",
        ")\n",
        "\n",
        "# Fit на всем датасете (unsupervised!)\n",
        "X_full = df[feature_cols].values\n",
        "X_full_scaled = scaler.fit_transform(X_full)\n",
        "\n",
        "iso_predictions = iso_forest.fit_predict(X_full_scaled)\n",
        "iso_scores = iso_forest.score_samples(X_full_scaled)  # anomaly scores\n",
        "\n",
        "# Convert: -1 (anomaly) → 1, 1 (normal) → 0\n",
        "iso_predictions_binary = (iso_predictions == -1).astype(int)\n",
        "\n",
        "print(f\"Isolation Forest Results:\")\n",
        "print(f\"  Predicted anomalies: {iso_predictions_binary.sum():,} ({iso_predictions_binary.mean()*100:.3f}%)\")\n",
        "print(f\"  True fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Evaluate Isolation Forest\n",
        "from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score\n",
        "\n",
        "y_true_full = df['Class'].values\n",
        "\n",
        "iso_precision = precision_score(y_true_full, iso_predictions_binary)\n",
        "iso_recall = recall_score(y_true_full, iso_predictions_binary)\n",
        "iso_f1 = f1_score(y_true_full, iso_predictions_binary)\n",
        "# Use anomaly scores for AUC (need to negate because lower score = more anomalous)\n",
        "iso_auc = roc_auc_score(y_true_full, -iso_scores)\n",
        "\n",
        "print(\"\\nIsolation Forest Performance:\")\n",
        "print(\"=\"*50)\n",
        "print(f\"  Precision: {iso_precision:.4f}\")\n",
        "print(f\"  Recall:    {iso_recall:.4f}\")\n",
        "print(f\"  F1 Score:  {iso_f1:.4f}\")\n",
        "print(f\"  ROC AUC:   {iso_auc:.4f}\")\n",
        "print(\"=\"*50)\n",
        "\n",
        "# Confusion matrix\n",
        "cm = confusion_matrix(y_true_full, iso_predictions_binary)\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n",
        "           xticklabels=['Normal', 'Fraud'],\n",
        "           yticklabels=['Normal', 'Fraud'])\n",
        "plt.title('Isolation Forest Confusion Matrix', fontsize=14, fontweight='bold')\n",
        "plt.ylabel('True')\n",
        "plt.xlabel('Predicted')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(f\"\\n📊 Interpretation:\")\n",
        "print(f\"  True Negatives (TN): {cm[0,0]:,} - correctly identified normal\")\n",
        "print(f\"  False Positives (FP): {cm[0,1]:,} - normal flagged as fraud\")\n",
        "print(f\"  False Negatives (FN): {cm[1,0]:,} - missed fraud\")\n",
        "print(f\"  True Positives (TP): {cm[1,1]:,} - correctly caught fraud\")"
    ]
})

# ============================================================================
# LOCAL OUTLIER FACTOR
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.2 Local Outlier Factor (LOF)\n",
        "\n",
        "**Принцип:** Density-based anomaly detection\n",
        "\n",
        "**Как работает:**\n",
        "1. Для каждой точки вычисляет локальную плотность (density)\n",
        "2. Сравнивает с плотностью соседей\n",
        "3. LOF > 1: точка в менее плотной области → outlier\n",
        "4. LOF ≈ 1: похожая плотность с соседями → normal\n",
        "\n",
        "**Преимущество:** Находит локальные outliers (даже если рядом с плотным кластером)"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# LOF (на sample из-за computational cost)\n",
        "sample_size_lof = 10000\n",
        "sample_indices = np.random.choice(len(X_full_scaled), sample_size_lof, replace=False)\n",
        "X_lof = X_full_scaled[sample_indices]\n",
        "y_lof = y_true_full[sample_indices]\n",
        "\n",
        "lof = LocalOutlierFactor(\n",
        "    n_neighbors=20,\n",
        "    contamination=0.002\n",
        ")\n",
        "\n",
        "lof_predictions = lof.fit_predict(X_lof)\n",
        "lof_scores = lof.negative_outlier_factor_  # higher = more normal\n",
        "\n",
        "# Convert: -1 → 1, 1 → 0\n",
        "lof_predictions_binary = (lof_predictions == -1).astype(int)\n",
        "\n",
        "print(f\"LOF Results (on {sample_size_lof:,} sample):\")\n",
        "print(f\"  Predicted anomalies: {lof_predictions_binary.sum():,}\")\n",
        "print(f\"  True fraud in sample: {y_lof.sum():,}\")\n",
        "\n",
        "# Evaluate\n",
        "lof_precision = precision_score(y_lof, lof_predictions_binary)\n",
        "lof_recall = recall_score(y_lof, lof_predictions_binary)\n",
        "lof_f1 = f1_score(y_lof, lof_predictions_binary)\n",
        "lof_auc = roc_auc_score(y_lof, -lof_scores)  # negate for AUC\n",
        "\n",
        "print(\"\\nLOF Performance:\")\n",
        "print(\"=\"*50)\n",
        "print(f\"  Precision: {lof_precision:.4f}\")\n",
        "print(f\"  Recall:    {lof_recall:.4f}\")\n",
        "print(f\"  F1 Score:  {lof_f1:.4f}\")\n",
        "print(f\"  ROC AUC:   {lof_auc:.4f}\")\n",
        "print(\"=\"*50)"
    ]
})

# ============================================================================
# ONE-CLASS SVM
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.3 One-Class SVM\n",
        "\n",
        "**Принцип:** Learn boundary of \"normal\" data\n",
        "\n",
        "**Как работает:**\n",
        "1. Fit hypersphere в feature space вокруг normal data\n",
        "2. Points outside hypersphere → anomalies\n",
        "3. nu parameter: expected fraction of anomalies\n",
        "\n",
        "**Преимущество:** Работает хорошо в high-dimensional spaces (kernel trick)"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# One-Class SVM (на sample)\n",
        "ocsvm = OneClassSVM(\n",
        "    kernel='rbf',\n",
        "    gamma='auto',\n",
        "    nu=0.002  # expected fraction of anomalies\n",
        ")\n",
        "\n",
        "ocsvm_predictions = ocsvm.fit_predict(X_lof)\n",
        "ocsvm_scores = ocsvm.score_samples(X_lof)\n",
        "\n",
        "# Convert: -1 → 1, 1 → 0\n",
        "ocsvm_predictions_binary = (ocsvm_predictions == -1).astype(int)\n",
        "\n",
        "print(f\"One-Class SVM Results:\")\n",
        "print(f\"  Predicted anomalies: {ocsvm_predictions_binary.sum():,}\")\n",
        "\n",
        "# Evaluate\n",
        "ocsvm_precision = precision_score(y_lof, ocsvm_predictions_binary)\n",
        "ocsvm_recall = recall_score(y_lof, ocsvm_predictions_binary)\n",
        "ocsvm_f1 = f1_score(y_lof, ocsvm_predictions_binary)\n",
        "ocsvm_auc = roc_auc_score(y_lof, -ocsvm_scores)\n",
        "\n",
        "print(\"\\nOne-Class SVM Performance:\")\n",
        "print(\"=\"*50)\n",
        "print(f\"  Precision: {ocsvm_precision:.4f}\")\n",
        "print(f\"  Recall:    {ocsvm_recall:.4f}\")\n",
        "print(f\"  F1 Score:  {ocsvm_f1:.4f}\")\n",
        "print(f\"  ROC AUC:   {ocsvm_auc:.4f}\")\n",
        "print(\"=\"*50)"
    ]
})

# ============================================================================
# COMPARISON
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.4 Method Comparison"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Compare all methods\n",
        "results = pd.DataFrame({\n",
        "    'Method': ['Isolation Forest', 'LOF', 'One-Class SVM'],\n",
        "    'Precision': [iso_precision, lof_precision, ocsvm_precision],\n",
        "    'Recall': [iso_recall, lof_recall, ocsvm_recall],\n",
        "    'F1 Score': [iso_f1, lof_f1, ocsvm_f1],\n",
        "    'ROC AUC': [iso_auc, lof_auc, ocsvm_auc]\n",
        "})\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"ANOMALY DETECTION METHODS COMPARISON\")\n",
        "print(\"=\"*70)\n",
        "print(results.to_string(index=False))\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Visualize\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# Precision/Recall/F1\n",
        "metrics = ['Precision', 'Recall', 'F1 Score']\n",
        "x = np.arange(len(results))\n",
        "width = 0.25\n",
        "\n",
        "for i, metric in enumerate(metrics):\n",
        "    axes[0].bar(x + i*width, results[metric], width, label=metric)\n",
        "\n",
        "axes[0].set_xlabel('Method')\n",
        "axes[0].set_ylabel('Score')\n",
        "axes[0].set_title('Precision, Recall, F1 Comparison', fontsize=14, fontweight='bold')\n",
        "axes[0].set_xticks(x + width)\n",
        "axes[0].set_xticklabels(results['Method'], rotation=15)\n",
        "axes[0].legend()\n",
        "axes[0].grid(alpha=0.3, axis='y')\n",
        "\n",
        "# ROC AUC\n",
        "axes[1].bar(results['Method'], results['ROC AUC'], color=['steelblue', 'orange', 'green'])\n",
        "axes[1].set_ylabel('ROC AUC')\n",
        "axes[1].set_title('ROC AUC Comparison', fontsize=14, fontweight='bold')\n",
        "axes[1].set_xticklabels(results['Method'], rotation=15)\n",
        "axes[1].grid(alpha=0.3, axis='y')\n",
        "for i, v in enumerate(results['ROC AUC']):\n",
        "    axes[1].text(i, v + 0.01, f\"{v:.3f}\", ha='center', fontweight='bold')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n📊 Key Findings:\")\n",
        "best_auc_idx = results['ROC AUC'].idxmax()\n",
        "best_f1_idx = results['F1 Score'].idxmax()\n",
        "print(f\"  Best ROC AUC: {results.loc[best_auc_idx, 'Method']} ({results.loc[best_auc_idx, 'ROC AUC']:.3f})\")\n",
        "print(f\"  Best F1: {results.loc[best_f1_idx, 'Method']} ({results.loc[best_f1_idx, 'F1 Score']:.3f})\")\n",
        "print(\"  All methods work reasonably well for unsupervised anomaly detection!\")\n",
        "print(\"  Isolation Forest typically best balance of speed and performance\")"
    ]
})

# Save
notebook['cells'] = cells
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Added clustering and anomaly detection: {len(cells)} cells total')
