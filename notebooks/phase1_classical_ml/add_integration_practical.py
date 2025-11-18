#!/usr/bin/env python3
"""
Добавление полной практической части в Integration notebook
Сравнение ВСЕХ методов Phase 1 на Telco Churn
"""

import json

# Читаем текущий ноутбук
notebook_path = '07_integration_comparison.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

practical_cells = []

# ============================================================================
# DATA LOADING
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.2 Загрузка и подготовка данных"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Загрузка Telco Churn данных\n",
        "import os\n",
        "\n",
        "data_path = '../../data/telco_churn.csv'\n",
        "\n",
        "if not os.path.exists(data_path):\n",
        "    print('❌ Файл не найден!')\n",
        "else:\n",
        "    df = pd.read_csv(data_path)\n",
        "    print(f'✅ Данные загружены: {df.shape[0]:,} строк, {df.shape[1]} столбцов')\n",
        "    print(f'Target: Churn')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Первый взгляд\n",
        "print('Первые строки:')\n",
        "df.head()"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# EDA: Class balance\n",
        "churn_counts = df['Churn'].value_counts()\n",
        "churn_pct = df['Churn'].value_counts(normalize=True) * 100\n",
        "\n",
        "print('Class distribution:')\n",
        "print(f'No: {churn_counts[\"No\"]} ({churn_pct[\"No\"]:.1f}%)')\n",
        "print(f'Yes: {churn_counts[\"Yes\"]} ({churn_pct[\"Yes\"]:.1f}%)')\n",
        "print(f'\\nImbalance ratio: 1:{churn_counts[\"No\"] / churn_counts[\"Yes\"]:.1f}')\n",
        "\n",
        "# Визуализация\n",
        "plt.figure(figsize=(8, 5))\n",
        "churn_counts.plot(kind='bar', color=['green', 'red'], alpha=0.7, edgecolor='black')\n",
        "plt.title('Class Distribution: Churn')\n",
        "plt.xlabel('Churn')\n",
        "plt.ylabel('Count')\n",
        "plt.xticks(rotation=0)\n",
        "for i, (idx, val) in enumerate(churn_counts.items()):\n",
        "    plt.text(i, val + 100, f'{val} ({churn_pct[idx]:.1f}%)', ha='center')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('\\n🔍 Moderate imbalance (~27% churn) → Class weights или moderate SMOTE')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Подготовка данных\n",
        "# Удаляем customerID (не нужен)\n",
        "if 'customerID' in df.columns:\n",
        "    df = df.drop('customerID', axis=1)\n",
        "\n",
        "# TotalCharges иногда имеет ' ' вместо чисел - исправляем\n",
        "if 'TotalCharges' in df.columns:\n",
        "    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\n",
        "    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)\n",
        "\n",
        "# Разделяем на numeric и categorical\n",
        "numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()\n",
        "categorical_features = df.select_dtypes(include=['object']).columns.tolist()\n",
        "categorical_features.remove('Churn')  # Target\n",
        "\n",
        "print(f'Numeric features ({len(numeric_features)}): {numeric_features}')\n",
        "print(f'Categorical features ({len(categorical_features)}): {categorical_features[:5]}...')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Target encoding\n",
        "y = (df['Churn'] == 'Yes').astype(int)\n",
        "\n",
        "# Features\n",
        "X = df.drop('Churn', axis=1)\n",
        "\n",
        "print(f'X shape: {X.shape}')\n",
        "print(f'y shape: {y.shape}')\n",
        "print(f'y distribution: {y.value_counts().to_dict()}')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Train/test split (stratified!)\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y\n",
        ")\n",
        "\n",
        "print(f'Train: {X_train.shape[0]:,} samples')\n",
        "print(f'Test: {X_test.shape[0]:,} samples')\n",
        "print(f'Train churn rate: {y_train.mean():.1%}')\n",
        "print(f'Test churn rate: {y_test.mean():.1%}')"
    ]
})

# ============================================================================
# PREPROCESSING VARIANTS
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.3 Подготовка разных версий данных\\n\",\n",
        "\n",
        "Создадим несколько версий для разных моделей:\n",
        "1. **One-hot encoded** — для Logistic Regression, XGBoost, LightGBM\n",
        "2. **Native categorical** — для CatBoost\n",
        "3. **Target encoded** — альтернативный вариант"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Version 1: One-hot encoding\n",
        "X_train_ohe = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)\n",
        "X_test_ohe = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)\n",
        "\n",
        "# Выровнять колонки\n",
        "X_train_ohe, X_test_ohe = X_train_ohe.align(X_test_ohe, join='left', axis=1, fill_value=0)\n",
        "\n",
        "print(f'One-hot encoded: {X_train_ohe.shape[1]} features')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Version 2: Label encoding для CatBoost (он поймет categorical сам)\n",
        "X_train_cat = X_train.copy()\n",
        "X_test_cat = X_test.copy()\n",
        "\n",
        "# CatBoost принимает categorical как индексы категорий\n",
        "cat_indices = []\n",
        "for i, col in enumerate(X_train_cat.columns):\n",
        "    if col in categorical_features:\n",
        "        le = LabelEncoder()\n",
        "        X_train_cat[col] = le.fit_transform(X_train_cat[col].astype(str))\n",
        "        X_test_cat[col] = le.transform(X_test_cat[col].astype(str))\n",
        "        cat_indices.append(i)\n",
        "\n",
        "print(f'CatBoost version: {len(cat_indices)} categorical features at indices {cat_indices[:5]}...')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Масштабирование для Logistic Regression\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train_ohe)\n",
        "X_test_scaled = scaler.transform(X_test_ohe)\n",
        "\n",
        "print('✅ Все версии данных подготовлены')"
    ]
})

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.4 Функция оценки моделей"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def evaluate_model(model, X_tr, X_te, y_tr, y_te, name, train_time=None):\n",
        "    \"\"\"\n",
        "    Комплексная оценка модели\n",
        "    \"\"\"\n",
        "    # Предсказания\n",
        "    y_pred = model.predict(X_te)\n",
        "    y_proba = model.predict_proba(X_te)[:, 1]\n",
        "    \n",
        "    # Метрики\n",
        "    acc = accuracy_score(y_te, y_pred)\n",
        "    prec = precision_score(y_te, y_pred)\n",
        "    rec = recall_score(y_te, y_pred)\n",
        "    f1 = f1_score(y_te, y_pred)\n",
        "    roc_auc = roc_auc_score(y_te, y_proba)\n",
        "    pr_auc = average_precision_score(y_te, y_proba)\n",
        "    \n",
        "    # Business cost (FN = $500, FP = $50)\n",
        "    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()\n",
        "    cost = fn * 500 + fp * 50\n",
        "    \n",
        "    # Вывод\n",
        "    print(f'\\n📊 {name}:')\n",
        "    print(f'  Accuracy: {acc:.4f}')\n",
        "    print(f'  Precision: {prec:.4f}')\n",
        "    print(f'  Recall: {rec:.4f}')\n",
        "    print(f'  F1-score: {f1:.4f}')\n",
        "    print(f'  ROC-AUC: {roc_auc:.4f}')\n",
        "    print(f'  PR-AUC: {pr_auc:.4f}')\n",
        "    print(f'  Business Cost: ${cost:,}')\n",
        "    if train_time:\n",
        "        print(f'  Train Time: {train_time:.2f}s')\n",
        "    print(f'  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}')\n",
        "    \n",
        "    return {\n",
        "        'Model': name,\n",
        "        'Accuracy': acc,\n",
        "        'Precision': prec,\n",
        "        'Recall': rec,\n",
        "        'F1': f1,\n",
        "        'ROC-AUC': roc_auc,\n",
        "        'PR-AUC': pr_auc,\n",
        "        'Cost': cost,\n",
        "        'Time': train_time if train_time else 0,\n",
        "        'TP': tp,\n",
        "        'FP': fp,\n",
        "        'TN': tn,\n",
        "        'FN': fn\n",
        "    }\n",
        "\n",
        "# Инициализируем результаты\n",
        "results = []\n",
        "\n",
        "print('✅ Функция оценки создана')"
    ]
})

# ============================================================================
# BASELINE
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.5 Baseline: Logistic Regression"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Logistic Regression baseline\n",
        "start = time.time()\n",
        "lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)\n",
        "lr.fit(X_train_scaled, y_train)\n",
        "lr_time = time.time() - start\n",
        "\n",
        "lr_results = evaluate_model(lr, X_train_scaled, X_test_scaled, y_train, y_test, \n",
        "                            'Logistic Regression', lr_time)\n",
        "results.append(lr_results)"
    ]
})

# ============================================================================
# XGBOOST
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.6 XGBoost: Baseline и Tuned"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# XGBoost baseline\n",
        "start = time.time()\n",
        "xgb_base = XGBClassifier(\n",
        "    n_estimators=100,\n",
        "    learning_rate=0.1,\n",
        "    max_depth=6,\n",
        "    random_state=RANDOM_STATE,\n",
        "    verbosity=0\n",
        ")\n",
        "xgb_base.fit(X_train_ohe, y_train)\n",
        "xgb_base_time = time.time() - start\n",
        "\n",
        "xgb_base_results = evaluate_model(xgb_base, X_train_ohe, X_test_ohe, y_train, y_test,\n",
        "                                  'XGBoost Baseline', xgb_base_time)\n",
        "results.append(xgb_base_results)"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# XGBoost tuned с class weights\n",
        "scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()\n",
        "\n",
        "start = time.time()\n",
        "xgb_tuned = XGBClassifier(\n",
        "    n_estimators=200,\n",
        "    learning_rate=0.05,\n",
        "    max_depth=4,\n",
        "    min_child_weight=3,\n",
        "    subsample=0.8,\n",
        "    colsample_bytree=0.8,\n",
        "    reg_alpha=0.1,\n",
        "    reg_lambda=1.0,\n",
        "    scale_pos_weight=scale_pos_weight,  # Class weights!\n",
        "    random_state=RANDOM_STATE,\n",
        "    verbosity=0\n",
        ")\n",
        "xgb_tuned.fit(X_train_ohe, y_train)\n",
        "xgb_tuned_time = time.time() - start\n",
        "\n",
        "xgb_tuned_results = evaluate_model(xgb_tuned, X_train_ohe, X_test_ohe, y_train, y_test,\n",
        "                                   'XGBoost Tuned + Weights', xgb_tuned_time)\n",
        "results.append(xgb_tuned_results)"
    ]
})

# ============================================================================
# LIGHTGBM
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.7 LightGBM: Baseline и Tuned"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# LightGBM baseline\n",
        "start = time.time()\n",
        "lgbm_base = LGBMClassifier(\n",
        "    n_estimators=100,\n",
        "    learning_rate=0.1,\n",
        "    num_leaves=31,\n",
        "    random_state=RANDOM_STATE,\n",
        "    verbose=-1\n",
        ")\n",
        "lgbm_base.fit(X_train_ohe, y_train)\n",
        "lgbm_base_time = time.time() - start\n",
        "\n",
        "lgbm_base_results = evaluate_model(lgbm_base, X_train_ohe, X_test_ohe, y_train, y_test,\n",
        "                                   'LightGBM Baseline', lgbm_base_time)\n",
        "results.append(lgbm_base_results)"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# LightGBM tuned с class weights\n",
        "start = time.time()\n",
        "lgbm_tuned = LGBMClassifier(\n",
        "    n_estimators=200,\n",
        "    learning_rate=0.05,\n",
        "    num_leaves=31,\n",
        "    max_depth=6,\n",
        "    min_child_samples=20,\n",
        "    subsample=0.8,\n",
        "    colsample_bytree=0.8,\n",
        "    reg_alpha=0.1,\n",
        "    reg_lambda=1.0,\n",
        "    scale_pos_weight=scale_pos_weight,\n",
        "    random_state=RANDOM_STATE,\n",
        "    verbose=-1\n",
        ")\n",
        "lgbm_tuned.fit(X_train_ohe, y_train)\n",
        "lgbm_tuned_time = time.time() - start\n",
        "\n",
        "lgbm_tuned_results = evaluate_model(lgbm_tuned, X_train_ohe, X_test_ohe, y_train, y_test,\n",
        "                                    'LightGBM Tuned + Weights', lgbm_tuned_time)\n",
        "results.append(lgbm_tuned_results)"
    ]
})

# ============================================================================
# CATBOOST
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.8 CatBoost: Native Categorical Features"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# CatBoost baseline (с native categorical)\n",
        "start = time.time()\n",
        "cat_base = CatBoostClassifier(\n",
        "    iterations=100,\n",
        "    learning_rate=0.1,\n",
        "    depth=6,\n",
        "    cat_features=cat_indices,  # NATIVE CATEGORICAL!\n",
        "    random_state=RANDOM_STATE,\n",
        "    verbose=0\n",
        ")\n",
        "cat_base.fit(X_train_cat, y_train)\n",
        "cat_base_time = time.time() - start\n",
        "\n",
        "cat_base_results = evaluate_model(cat_base, X_train_cat, X_test_cat, y_train, y_test,\n",
        "                                  'CatBoost Baseline (Native Cat)', cat_base_time)\n",
        "results.append(cat_base_results)"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# CatBoost tuned с class weights\n",
        "start = time.time()\n",
        "cat_tuned = CatBoostClassifier(\n",
        "    iterations=200,\n",
        "    learning_rate=0.05,\n",
        "    depth=6,\n",
        "    l2_leaf_reg=3,\n",
        "    cat_features=cat_indices,\n",
        "    class_weights=[1, scale_pos_weight],  # Class weights!\n",
        "    random_state=RANDOM_STATE,\n",
        "    verbose=0\n",
        ")\n",
        "cat_tuned.fit(X_train_cat, y_train)\n",
        "cat_tuned_time = time.time() - start\n",
        "\n",
        "cat_tuned_results = evaluate_model(cat_tuned, X_train_cat, X_test_cat, y_train, y_test,\n",
        "                                   'CatBoost Tuned + Weights', cat_tuned_time)\n",
        "results.append(cat_tuned_results)"
    ]
})

# ============================================================================
# SMOTE
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.9 Imbalanced Data: SMOTE"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# SMOTE + XGBoost\n",
        "if HAS_IMBLEARN:\n",
        "    smote = SMOTE(sampling_strategy=0.7, random_state=RANDOM_STATE)  # До 70% minority\n",
        "    X_train_smote, y_train_smote = smote.fit_resample(X_train_ohe, y_train)\n",
        "    \n",
        "    print(f'После SMOTE: {X_train_smote.shape[0]} samples')\n",
        "    print(f'Churn rate: {y_train_smote.mean():.1%}')\n",
        "    \n",
        "    start = time.time()\n",
        "    xgb_smote = XGBClassifier(\n",
        "        n_estimators=200,\n",
        "        learning_rate=0.05,\n",
        "        max_depth=4,\n",
        "        subsample=0.8,\n",
        "        colsample_bytree=0.8,\n",
        "        random_state=RANDOM_STATE,\n",
        "        verbosity=0\n",
        "    )\n",
        "    xgb_smote.fit(X_train_smote, y_train_smote)\n",
        "    xgb_smote_time = time.time() - start\n",
        "    \n",
        "    xgb_smote_results = evaluate_model(xgb_smote, X_train_smote, X_test_ohe, \n",
        "                                       y_train_smote, y_test,\n",
        "                                       'XGBoost + SMOTE', xgb_smote_time)\n",
        "    results.append(xgb_smote_results)\n",
        "else:\n",
        "    print('⚠️ SMOTE пропущен (imbalanced-learn не установлен)')"
    ]
})

# ============================================================================
# STACKING
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.10 Stacking Ensemble"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Stacking: XGBoost + LightGBM + CatBoost → Logistic Regression\n",
        "print('🚀 Создаем Stacking Ensemble...')\n",
        "\n",
        "# Out-of-fold predictions для meta features\n",
        "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)\n",
        "\n",
        "# Базовые модели (уже обучены)\n",
        "base_models = [\n",
        "    ('xgb', xgb_tuned, X_train_ohe, X_test_ohe),\n",
        "    ('lgbm', lgbm_tuned, X_train_ohe, X_test_ohe),\n",
        "    ('cat', cat_tuned, X_train_cat, X_test_cat)\n",
        "]\n",
        "\n",
        "# OOF predictions\n",
        "oof_train = np.zeros((X_train.shape[0], len(base_models)))\n",
        "oof_test = np.zeros((X_test.shape[0], len(base_models)))\n",
        "\n",
        "for i, (name, model, X_tr, X_te) in enumerate(base_models):\n",
        "    # Используем уже обученные модели для test\n",
        "    oof_test[:, i] = model.predict_proba(X_te)[:, 1]\n",
        "    \n",
        "    # OOF для train через cross-validation\n",
        "    oof_preds = cross_val_predict(model, X_tr, y_train, cv=cv, method='predict_proba')[:, 1]\n",
        "    oof_train[:, i] = oof_preds\n",
        "    \n",
        "    print(f'✅ {name}: OOF готов')\n",
        "\n",
        "print(f'Meta features shape: train {oof_train.shape}, test {oof_test.shape}')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Meta-learner: Logistic Regression\n",
        "start = time.time()\n",
        "meta_learner = LogisticRegression(random_state=RANDOM_STATE)\n",
        "meta_learner.fit(oof_train, y_train)\n",
        "stacking_time = time.time() - start\n",
        "\n",
        "# Предсказания\n",
        "stacking_pred_proba = meta_learner.predict_proba(oof_test)[:, 1]\n",
        "stacking_pred = (stacking_pred_proba > 0.5).astype(int)\n",
        "\n",
        "# Оценка\n",
        "acc = accuracy_score(y_test, stacking_pred)\n",
        "prec = precision_score(y_test, stacking_pred)\n",
        "rec = recall_score(y_test, stacking_pred)\n",
        "f1 = f1_score(y_test, stacking_pred)\n",
        "roc_auc = roc_auc_score(y_test, stacking_pred_proba)\n",
        "pr_auc = average_precision_score(y_test, stacking_pred_proba)\n",
        "\n",
        "tn, fp, fn, tp = confusion_matrix(y_test, stacking_pred).ravel()\n",
        "cost = fn * 500 + fp * 50\n",
        "\n",
        "print(f'\\n📊 Stacking Ensemble:')\n",
        "print(f'  Accuracy: {acc:.4f}')\n",
        "print(f'  Precision: {prec:.4f}')\n",
        "print(f'  Recall: {rec:.4f}')\n",
        "print(f'  F1-score: {f1:.4f}')\n",
        "print(f'  ROC-AUC: {roc_auc:.4f}')\n",
        "print(f'  PR-AUC: {pr_auc:.4f}')\n",
        "print(f'  Business Cost: ${cost:,}')\n",
        "print(f'  Meta-learner weights: {meta_learner.coef_[0]}')\n",
        "\n",
        "stacking_results = {\n",
        "    'Model': 'Stacking Ensemble',\n",
        "    'Accuracy': acc,\n",
        "    'Precision': prec,\n",
        "    'Recall': rec,\n",
        "    'F1': f1,\n",
        "    'ROC-AUC': roc_auc,\n",
        "    'PR-AUC': pr_auc,\n",
        "    'Cost': cost,\n",
        "    'Time': stacking_time,\n",
        "    'TP': tp,\n",
        "    'FP': fp,\n",
        "    'TN': tn,\n",
        "    'FN': fn\n",
        "}\n",
        "results.append(stacking_results)"
    ]
})

# ============================================================================
# FINAL COMPARISON
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.11 Финальное сравнение всех методов"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Создаем сравнительную таблицу\n",
        "comparison_df = pd.DataFrame(results)\n",
        "comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)\n",
        "\n",
        "print('\\n' + '='*100)\n",
        "print('🏆 ФИНАЛЬНОЕ СРАВНЕНИЕ ВСЕХ МЕТОДОВ PHASE 1')\n",
        "print('='*100)\n",
        "print(comparison_df[['Model', 'ROC-AUC', 'F1', 'Precision', 'Recall', 'Cost', 'Time']].to_string(index=False))\n",
        "print('='*100)"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Визуализация сравнения\n",
        "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
        "\n",
        "# ROC-AUC\n",
        "ax = axes[0, 0]\n",
        "bars = ax.barh(comparison_df['Model'], comparison_df['ROC-AUC'], color='skyblue', edgecolor='black')\n",
        "ax.set_xlabel('ROC-AUC')\n",
        "ax.set_title('ROC-AUC Score')\n",
        "ax.axvline(comparison_df['ROC-AUC'].max(), color='red', linestyle='--', alpha=0.5)\n",
        "\n",
        "# F1-score\n",
        "ax = axes[0, 1]\n",
        "ax.barh(comparison_df['Model'], comparison_df['F1'], color='lightgreen', edgecolor='black')\n",
        "ax.set_xlabel('F1-score')\n",
        "ax.set_title('F1 Score')\n",
        "\n",
        "# Precision vs Recall\n",
        "ax = axes[0, 2]\n",
        "ax.scatter(comparison_df['Recall'], comparison_df['Precision'], s=200, alpha=0.6, edgecolor='black')\n",
        "for i, model in enumerate(comparison_df['Model']):\n",
        "    ax.annotate(model.split()[0], \n",
        "                (comparison_df['Recall'].iloc[i], comparison_df['Precision'].iloc[i]),\n",
        "                fontsize=8)\n",
        "ax.set_xlabel('Recall')\n",
        "ax.set_ylabel('Precision')\n",
        "ax.set_title('Precision vs Recall')\n",
        "ax.grid(True, alpha=0.3)\n",
        "\n",
        "# Business Cost\n",
        "ax = axes[1, 0]\n",
        "comparison_sorted_cost = comparison_df.sort_values('Cost')\n",
        "ax.barh(comparison_sorted_cost['Model'], comparison_sorted_cost['Cost'], \n",
        "        color='lightcoral', edgecolor='black')\n",
        "ax.set_xlabel('Business Cost ($)')\n",
        "ax.set_title('Business Cost (Lower is Better)')\n",
        "\n",
        "# Training Time\n",
        "ax = axes[1, 1]\n",
        "comparison_sorted_time = comparison_df.sort_values('Time')\n",
        "ax.barh(comparison_sorted_time['Model'], comparison_sorted_time['Time'], \n",
        "        color='wheat', edgecolor='black')\n",
        "ax.set_xlabel('Time (seconds)')\n",
        "ax.set_title('Training Time')\n",
        "\n",
        "# Confusion Matrix for best model\n",
        "ax = axes[1, 2]\n",
        "best_model = comparison_df.iloc[0]\n",
        "cm = np.array([[best_model['TN'], best_model['FP']], \n",
        "               [best_model['FN'], best_model['TP']]])\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)\n",
        "ax.set_xlabel('Predicted')\n",
        "ax.set_ylabel('Actual')\n",
        "ax.set_title(f'Best Model: {best_model[\"Model\"]}')\n",
        "ax.set_xticklabels(['No Churn', 'Churn'])\n",
        "ax.set_yticklabels(['No Churn', 'Churn'])\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# ============================================================================
# CONCLUSIONS
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🎯 Выводы и Рекомендации\n",
        "\n",
        "### Что мы выяснили:\n",
        "\n",
        "#### 1. Качество моделей (ROC-AUC)\n",
        "\n",
        "Типичные результаты (зависит от данных):\n",
        "- **Stacking Ensemble:** Обычно лучший (~0.85-0.87)\n",
        "- **CatBoost Tuned:** Близко к stacking (~0.84-0.86)\n",
        "- **XGBoost/LightGBM Tuned:** Сопоставимо (~0.83-0.85)\n",
        "- **Базовые boosting:** Хорошо (~0.81-0.83)\n",
        "- **Logistic Regression:** Baseline (~0.75-0.78)\n",
        "\n",
        "**Вывод:** Разница между tuned моделями часто <2% ROC-AUC\n",
        "\n",
        "#### 2. Скорость обучения\n",
        "\n",
        "- **LightGBM:** Самый быстрый (особенно на больших данных)\n",
        "- **XGBoost:** Средняя скорость\n",
        "- **CatBoost:** Медленнее (ordered boosting сложнее)\n",
        "- **Stacking:** Самый медленный (обучение всех моделей)\n",
        "\n",
        "#### 3. Categorical features\n",
        "\n",
        "**CatBoost с native categorical показывает отличные результаты БЕЗ:**\n",
        "- One-hot encoding\n",
        "- Target encoding\n",
        "- Feature engineering\n",
        "\n",
        "**Вывод:** Для датасетов с категориальными - CatBoost лучший выбор из коробки!\n",
        "\n",
        "#### 4. Imbalanced data\n",
        "\n",
        "**Class weights vs SMOTE:**\n",
        "- Class weights: Проще, быстрее, работает хорошо для moderate imbalance\n",
        "- SMOTE: Может помочь для extreme imbalance, но риск noise\n",
        "- **Hybrid** (moderate SMOTE + weights): Часто лучший вариант\n",
        "\n",
        "#### 5. Stacking\n",
        "\n",
        "**Прирост:** Обычно 0.5-2% поверх лучшей базовой модели\n",
        "\n",
        "**Стоит ли?**\n",
        "- ✅ Kaggle, research: Да!\n",
        "- ❌ Production с ограничениями: Вероятно нет\n",
        "\n",
        "---\n",
        "\n",
        "### Практические рекомендации для Production\n",
        "\n",
        "#### Сценарий 1: Быстрый старт\n",
        "\n",
        "**Рекомендация:** XGBoost с дефолтными параметрами\n",
        "\n",
        "```python\n",
        "XGBClassifier(n_estimators=100, learning_rate=0.1)\n",
        "```\n",
        "\n",
        "- ✅ Хорошее качество из коробки\n",
        "- ✅ Стабильность\n",
        "- ✅ Минимальный tuning\n",
        "\n",
        "#### Сценарий 2: Много категориальных признаков\n",
        "\n",
        "**Рекомендация:** CatBoost с native categorical\n",
        "\n",
        "```python\n",
        "CatBoostClassifier(iterations=100, cat_features=cat_indices)\n",
        "```\n",
        "\n",
        "- ✅ Не нужен one-hot encoding\n",
        "- ✅ Отличное качество по умолчанию\n",
        "- ✅ Ordered target statistics автоматически\n",
        "\n",
        "#### Сценарий 3: Большие данные (>1M строк)\n",
        "\n",
        "**Рекомендация:** LightGBM\n",
        "\n",
        "```python\n",
        "LGBMClassifier(n_estimators=100, num_leaves=31)\n",
        "```\n",
        "\n",
        "- ✅ В 5-20x быстрее XGBoost\n",
        "- ✅ Меньше памяти\n",
        "- ⚠️ Требует tuning для избежания overfitting\n",
        "\n",
        "#### Сценарий 4: Максимальное качество (Kaggle)\n",
        "\n",
        "**Рекомендация:** Stacking Ensemble\n",
        "\n",
        "```python\n",
        "Base: XGBoost + LightGBM + CatBoost + RF\n",
        "Meta: LogisticRegression / XGBoost\n",
        "```\n",
        "\n",
        "- ✅ Выжимаем последние проценты\n",
        "- ✅ Diversity базовых моделей\n",
        "- ❌ Сложность и время обучения\n",
        "\n",
        "#### Сценарий 5: Imbalanced data\n",
        "\n",
        "**Moderate imbalance (1:10):**\n",
        "```python\n",
        "scale_pos_weight = n_negative / n_positive\n",
        "XGBClassifier(scale_pos_weight=scale_pos_weight)\n",
        "```\n",
        "\n",
        "**Extreme imbalance (1:100+):**\n",
        "```python\n",
        "smote = SMOTE(sampling_strategy=0.1)  # Moderate oversample\n",
        "+ class_weights\n",
        "```\n",
        "\n",
        "---\n",
        "\n",
        "### Общие Best Practices\n",
        "\n",
        "1. **Всегда начинайте с baseline** (Logistic Regression)\n",
        "2. **Используйте cross-validation** для честной оценки\n",
        "3. **Class imbalance:** Используйте PR-AUC, не accuracy!\n",
        "4. **Feature engineering:** 60-70% качества, инвестируйте время\n",
        "5. **Hyperparameter tuning:** Только после feature engineering\n",
        "6. **Ensemble:** Последний шаг, если нужны проценты\n",
        "7. **Production:** Простота > сложность (XGBoost или CatBoost)\n",
        "\n",
        "---\n",
        "\n",
        "## 🎉 Phase 1 Classical ML завершен!\n",
        "\n",
        "**Мы изучили:**\n",
        "- ✅ XGBoost, LightGBM, CatBoost в деталях\n",
        "- ✅ Stacking Ensemble\n",
        "- ✅ Imbalanced data техники\n",
        "- ✅ Advanced Feature Engineering\n",
        "- ✅ Практическое сравнение на реальных данных\n",
        "\n",
        "**Следующие шаги:**\n",
        "- Phase 2: Deep Learning (MLP, CNN, RNN, Transformers)\n",
        "- Phase 3: Time Series\n",
        "- Phase 4: NLP\n",
        "- Phase 5: Computer Vision\n",
        "\n",
        "**Поздравляю! 🚀**\n"
    ]
})

# Добавляем практические ячейки
for cell in practical_cells:
    notebook['cells'].append(cell)

# Сохраняем
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Добавлено {len(practical_cells)} практических ячеек')
print(f'Всего ячеек в ноутбуке: {len(notebook["cells"])}')
