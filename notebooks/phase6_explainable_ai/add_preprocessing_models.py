#!/usr/bin/env python3
"""
Phase 6: Explainable AI - Part 2
Add: EDA, Preprocessing, Model Training
"""

import json

# Загружаем существующий notebook
notebook_path = '/home/user/test/notebooks/phase6_explainable_ai/01_explainable_ai_xai.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

cells = notebook['cells']

# ============================================================================
# EDA
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.3 Exploratory Data Analysis (EDA)\n",
        "\n",
        "Проверим распределение признаков и target variable, особенно для **fairness analysis**."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Базовая статистика\n",
        "print(\"=\" * 60)\n",
        "print(\"ИНФОРМАЦИЯ О ДАТАСЕТЕ\")\n",
        "print(\"=\" * 60)\n",
        "print(df.info())\n",
        "print(\"\\n\" + \"=\" * 60)\n",
        "print(\"СТАТИСТИКА ЧИСЛОВЫХ ПРИЗНАКОВ\")\n",
        "print(\"=\" * 60)\n",
        "print(df.describe())\n",
        "\n",
        "# Распределение target\n",
        "print(\"\\n\" + \"=\" * 60)\n",
        "print(\"РАСПРЕДЕЛЕНИЕ INCOME\")\n",
        "print(\"=\" * 60)\n",
        "print(df['income'].value_counts())\n",
        "print(f\"\\nClass balance: {df['income'].value_counts(normalize=True).to_dict()}\")\n"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Визуализация распределений для fairness analysis\n",
        "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
        "\n",
        "# Income by Sex (FAIRNESS CONCERN!)\n",
        "pd.crosstab(df['sex'], df['income'], normalize='index').plot(\n",
        "    kind='bar', ax=axes[0, 0], color=['#e74c3c', '#2ecc71']\n",
        ")\n",
        "axes[0, 0].set_title('Income Distribution by Sex', fontsize=14, fontweight='bold')\n",
        "axes[0, 0].set_xlabel('Sex')\n",
        "axes[0, 0].set_ylabel('Proportion')\n",
        "axes[0, 0].legend(title='Income')\n",
        "axes[0, 0].tick_params(axis='x', rotation=0)\n",
        "\n",
        "# Income by Race (FAIRNESS CONCERN!)\n",
        "pd.crosstab(df['race'], df['income'], normalize='index').plot(\n",
        "    kind='bar', ax=axes[0, 1], color=['#e74c3c', '#2ecc71']\n",
        ")\n",
        "axes[0, 1].set_title('Income Distribution by Race', fontsize=14, fontweight='bold')\n",
        "axes[0, 1].set_xlabel('Race')\n",
        "axes[0, 1].set_ylabel('Proportion')\n",
        "axes[0, 1].legend(title='Income')\n",
        "\n",
        "# Income by Education\n",
        "pd.crosstab(df['education'], df['income'], normalize='index').plot(\n",
        "    kind='bar', ax=axes[0, 2], color=['#e74c3c', '#2ecc71']\n",
        ")\n",
        "axes[0, 2].set_title('Income Distribution by Education', fontsize=14, fontweight='bold')\n",
        "axes[0, 2].set_xlabel('Education')\n",
        "axes[0, 2].set_ylabel('Proportion')\n",
        "axes[0, 2].legend(title='Income')\n",
        "\n",
        "# Age distribution\n",
        "df[df['income'] == '<=50K']['age'].hist(bins=30, alpha=0.5, label='<=50K', ax=axes[1, 0], color='#e74c3c')\n",
        "df[df['income'] == '>50K']['age'].hist(bins=30, alpha=0.5, label='>50K', ax=axes[1, 0], color='#2ecc71')\n",
        "axes[1, 0].set_title('Age Distribution by Income', fontsize=14, fontweight='bold')\n",
        "axes[1, 0].set_xlabel('Age')\n",
        "axes[1, 0].set_ylabel('Frequency')\n",
        "axes[1, 0].legend()\n",
        "\n",
        "# Hours per week\n",
        "df[df['income'] == '<=50K']['hours-per-week'].hist(bins=30, alpha=0.5, label='<=50K', ax=axes[1, 1], color='#e74c3c')\n",
        "df[df['income'] == '>50K']['hours-per-week'].hist(bins=30, alpha=0.5, label='>50K', ax=axes[1, 1], color='#2ecc71')\n",
        "axes[1, 1].set_title('Hours per Week by Income', fontsize=14, fontweight='bold')\n",
        "axes[1, 1].set_xlabel('Hours per Week')\n",
        "axes[1, 1].set_ylabel('Frequency')\n",
        "axes[1, 1].legend()\n",
        "\n",
        "# Education years\n",
        "df.boxplot(column='education-num', by='income', ax=axes[1, 2])\n",
        "axes[1, 2].set_title('Education Years by Income', fontsize=14, fontweight='bold')\n",
        "axes[1, 2].set_xlabel('Income')\n",
        "axes[1, 2].set_ylabel('Education Years')\n",
        "plt.sca(axes[1, 2])\n",
        "plt.xticks(rotation=0)\n",
        "\n",
        "plt.suptitle('EDA: Income Distribution Analysis', fontsize=16, fontweight='bold', y=1.00)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n⚠️ FAIRNESS CONCERNS DETECTED:\")\n",
        "print(\"- Income distribution varies by SEX\")\n",
        "print(\"- Income distribution varies by RACE\")\n",
        "print(\"→ Нужен fairness analysis для проверки модели на bias!\")\n"
    ]
})

# ============================================================================
# PREPROCESSING
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.4 Preprocessing\n",
        "\n",
        "**Задачи:**\n",
        "1. Encoding категориальных признаков\n",
        "2. Scaling числовых признаков\n",
        "3. Train/Test split\n",
        "4. **Важно:** Сохраним sensitive attributes (sex, race) отдельно для fairness analysis"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Создаём копию для preprocessing\n",
        "df_processed = df.copy()\n",
        "\n",
        "# Бинарный target\n",
        "df_processed['income'] = (df_processed['income'] == '>50K').astype(int)\n",
        "\n",
        "# Сохраняем sensitive attributes ДО encoding (для fairness analysis)\n",
        "sensitive_features = df_processed[['sex', 'race']].copy()\n",
        "\n",
        "# Определяем типы признаков\n",
        "categorical_cols = [\n",
        "    'workclass', 'education', 'marital-status', 'occupation',\n",
        "    'relationship', 'race', 'sex', 'native-country'\n",
        "]\n",
        "\n",
        "numerical_cols = [\n",
        "    'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'\n",
        "]\n",
        "\n",
        "# Label Encoding для категориальных признаков\n",
        "label_encoders = {}\n",
        "for col in categorical_cols:\n",
        "    le = LabelEncoder()\n",
        "    df_processed[col] = le.fit_transform(df_processed[col])\n",
        "    label_encoders[col] = le\n",
        "\n",
        "print(\"✅ Label Encoding completed\")\n",
        "print(f\"Категориальных признаков: {len(categorical_cols)}\")\n",
        "print(f\"Числовых признаков: {len(numerical_cols)}\")\n",
        "\n",
        "# Feature matrix и target\n",
        "feature_cols = categorical_cols + numerical_cols\n",
        "X = df_processed[feature_cols]\n",
        "y = df_processed['income']\n",
        "\n",
        "print(f\"\\nFeature matrix shape: {X.shape}\")\n",
        "print(f\"Target shape: {y.shape}\")\n",
        "print(f\"Positive class ratio: {y.mean():.3f}\")\n"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Train/Test split\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=42, stratify=y\n",
        ")\n",
        "\n",
        "# Также split для sensitive features (нужно для fairness analysis)\n",
        "sensitive_train, sensitive_test = train_test_split(\n",
        "    sensitive_features, test_size=0.2, random_state=42, stratify=y\n",
        ")\n",
        "\n",
        "# Scaling только для числовых признаков\n",
        "scaler = StandardScaler()\n",
        "\n",
        "# Важно: scaling только на train, затем transform на test\n",
        "X_train_scaled = X_train.copy()\n",
        "X_test_scaled = X_test.copy()\n",
        "\n",
        "X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])\n",
        "X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])\n",
        "\n",
        "print(\"✅ Train/Test split completed\")\n",
        "print(f\"Train size: {X_train.shape[0]} ({X_train.shape[0] / len(X):.1%})\")\n",
        "print(f\"Test size: {X_test.shape[0]} ({X_test.shape[0] / len(X):.1%})\")\n",
        "print(f\"\\nTrain positive class: {y_train.mean():.3f}\")\n",
        "print(f\"Test positive class: {y_test.mean():.3f}\")\n"
    ]
})

# ============================================================================
# MODEL TRAINING
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.5 Model Training\n",
        "\n",
        "Обучим 3 модели разной complexity для сравнения interpretability:\n",
        "\n",
        "1. **Logistic Regression** - простая, линейная, interpretable из коробки\n",
        "2. **Random Forest** - ensemble, нелинейная, средней сложности\n",
        "3. **XGBoost** - gradient boosting, высокая точность, сложная\n",
        "\n",
        "Затем применим XAI методы ко всем трём."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Dictionary для хранения моделей и результатов\n",
        "models = {}\n",
        "results = {}\n",
        "\n",
        "print(\"=\" * 70)\n",
        "print(\"ОБУЧЕНИЕ МОДЕЛЕЙ\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "# 1. Logistic Regression\n",
        "print(\"\\n[1/3] Training Logistic Regression...\")\n",
        "lr = LogisticRegression(max_iter=1000, random_state=42)\n",
        "lr.fit(X_train_scaled, y_train)\n",
        "lr_pred = lr.predict(X_test_scaled)\n",
        "lr_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]\n",
        "\n",
        "models['Logistic Regression'] = lr\n",
        "results['Logistic Regression'] = {\n",
        "    'predictions': lr_pred,\n",
        "    'probabilities': lr_pred_proba,\n",
        "    'accuracy': accuracy_score(y_test, lr_pred),\n",
        "    'precision': precision_score(y_test, lr_pred),\n",
        "    'recall': recall_score(y_test, lr_pred),\n",
        "    'f1': f1_score(y_test, lr_pred),\n",
        "    'roc_auc': roc_auc_score(y_test, lr_pred_proba)\n",
        "}\n",
        "print(\"✅ Logistic Regression trained\")\n",
        "\n",
        "# 2. Random Forest\n",
        "print(\"\\n[2/3] Training Random Forest...\")\n",
        "rf = RandomForestClassifier(\n",
        "    n_estimators=100,\n",
        "    max_depth=10,\n",
        "    min_samples_split=10,\n",
        "    random_state=42,\n",
        "    n_jobs=-1\n",
        ")\n",
        "rf.fit(X_train_scaled, y_train)\n",
        "rf_pred = rf.predict(X_test_scaled)\n",
        "rf_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]\n",
        "\n",
        "models['Random Forest'] = rf\n",
        "results['Random Forest'] = {\n",
        "    'predictions': rf_pred,\n",
        "    'probabilities': rf_pred_proba,\n",
        "    'accuracy': accuracy_score(y_test, rf_pred),\n",
        "    'precision': precision_score(y_test, rf_pred),\n",
        "    'recall': recall_score(y_test, rf_pred),\n",
        "    'f1': f1_score(y_test, rf_pred),\n",
        "    'roc_auc': roc_auc_score(y_test, rf_pred_proba)\n",
        "}\n",
        "print(\"✅ Random Forest trained\")\n",
        "\n",
        "# 3. XGBoost\n",
        "print(\"\\n[3/3] Training XGBoost...\")\n",
        "xgb_model = xgb.XGBClassifier(\n",
        "    n_estimators=100,\n",
        "    max_depth=6,\n",
        "    learning_rate=0.1,\n",
        "    random_state=42,\n",
        "    n_jobs=-1,\n",
        "    eval_metric='logloss'\n",
        ")\n",
        "xgb_model.fit(X_train_scaled, y_train)\n",
        "xgb_pred = xgb_model.predict(X_test_scaled)\n",
        "xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]\n",
        "\n",
        "models['XGBoost'] = xgb_model\n",
        "results['XGBoost'] = {\n",
        "    'predictions': xgb_pred,\n",
        "    'probabilities': xgb_pred_proba,\n",
        "    'accuracy': accuracy_score(y_test, xgb_pred),\n",
        "    'precision': precision_score(y_test, xgb_pred),\n",
        "    'recall': recall_score(y_test, xgb_pred),\n",
        "    'f1': f1_score(y_test, xgb_pred),\n",
        "    'roc_auc': roc_auc_score(y_test, xgb_pred_proba)\n",
        "}\n",
        "print(\"✅ XGBoost trained\")\n",
        "\n",
        "print(\"\\n\" + \"=\" * 70)\n",
        "print(\"✅ ВСЕ МОДЕЛИ ОБУЧЕНЫ\")\n",
        "print(\"=\" * 70)\n"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Сравнение моделей\n",
        "comparison_df = pd.DataFrame({\n",
        "    model_name: {\n",
        "        'Accuracy': metrics['accuracy'],\n",
        "        'Precision': metrics['precision'],\n",
        "        'Recall': metrics['recall'],\n",
        "        'F1-Score': metrics['f1'],\n",
        "        'ROC AUC': metrics['roc_auc']\n",
        "    }\n",
        "    for model_name, metrics in results.items()\n",
        "}).T\n",
        "\n",
        "print(\"\\n\" + \"=\" * 70)\n",
        "print(\"MODEL COMPARISON\")\n",
        "print(\"=\" * 70)\n",
        "print(comparison_df.round(4))\n",
        "\n",
        "# Визуализация сравнения\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
        "\n",
        "# Metrics comparison\n",
        "comparison_df.plot(kind='bar', ax=axes[0], width=0.8)\n",
        "axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')\n",
        "axes[0].set_xlabel('Model')\n",
        "axes[0].set_ylabel('Score')\n",
        "axes[0].legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')\n",
        "axes[0].set_ylim([0, 1])\n",
        "axes[0].tick_params(axis='x', rotation=0)\n",
        "axes[0].grid(axis='y', alpha=0.3)\n",
        "\n",
        "# ROC Curves\n",
        "for model_name, metrics in results.items():\n",
        "    fpr, tpr, _ = roc_curve(y_test, metrics['probabilities'])\n",
        "    auc = metrics['roc_auc']\n",
        "    axes[1].plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)\n",
        "\n",
        "axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)\n",
        "axes[1].set_title('ROC Curves', fontsize=14, fontweight='bold')\n",
        "axes[1].set_xlabel('False Positive Rate')\n",
        "axes[1].set_ylabel('True Positive Rate')\n",
        "axes[1].legend(loc='lower right')\n",
        "axes[1].grid(alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n🎯 Наблюдения:\")\n",
        "print(\"- XGBoost показывает лучшую performance (обычно)\")\n",
        "print(\"- Random Forest близок к XGBoost\")\n",
        "print(\"- Logistic Regression проще, но менее точная\")\n",
        "print(\"\\n→ Теперь применим XAI методы для интерпретации!\")\n"
    ]
})

# Сохраняем обновленный notebook
notebook['cells'] = cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'\\n✅ Updated notebook: {notebook_path}')
print(f'Total cells: {len(cells)}')
print('Preprocessing and models added!')
