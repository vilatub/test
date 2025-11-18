#!/usr/bin/env python3
"""
Phase 6: Explainable AI - Part 3
Add: SHAP Analysis (TreeSHAP for RF and XGBoost)
"""

import json

# Загружаем существующий notebook
notebook_path = '/home/user/test/notebooks/phase6_explainable_ai/01_explainable_ai_xai.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

cells = notebook['cells']

# ============================================================================
# SHAP INTRODUCTION
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## 💡 Часть 2: SHAP Analysis\n",
        "\n",
        "### Что такое SHAP?\n",
        "\n",
        "**SHAP (SHapley Additive exPlanations)** - unified framework для объяснения predictions любых ML моделей.\n",
        "\n",
        "**Основан на:**\n",
        "- **Shapley Values** из теории игр (Lloyd Shapley, Nobel Prize 2012)\n",
        "- **Идея:** Сколько каждый признак \"вкладывает\" в предсказание?\n",
        "\n",
        "**Почему SHAP лучше других методов:**\n",
        "\n",
        "1. **Теоретически обоснован** - единственный метод с гарантиями:\n",
        "   - ✅ **Local accuracy:** sum(SHAP values) = prediction - baseline\n",
        "   - ✅ **Consistency:** если признак важнее в модели B, его SHAP value выше\n",
        "   - ✅ **Missingness:** отсутствующий признак имеет SHAP = 0\n",
        "\n",
        "2. **Универсальность:**\n",
        "   - `TreeSHAP`: для древовидных моделей (RF, XGBoost, LightGBM) - **очень быстро**\n",
        "   - `KernelSHAP`: для любых моделей (черный ящик) - медленнее\n",
        "   - `DeepSHAP`: для нейронных сетей\n",
        "   - `LinearSHAP`: для линейных моделей\n",
        "\n",
        "3. **Global + Local:**\n",
        "   - Global importance: среднее |SHAP value| по всем samples\n",
        "   - Local explanation: SHAP values для конкретного prediction\n",
        "\n",
        "**Интерпретация SHAP value:**\n",
        "- `SHAP value > 0`: признак увеличивает prediction (пушит к классу 1)\n",
        "- `SHAP value < 0`: признак уменьшает prediction (пушит к классу 0)\n",
        "- `|SHAP value|` = magnitude of effect\n",
        "\n",
        "---\n"
    ]
})

# ============================================================================
# SHAP FOR RANDOM FOREST
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.1 SHAP для Random Forest\n",
        "\n",
        "Используем **TreeSHAP** - точный и быстрый алгоритм для древовидных моделей."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"=\" * 70)\n",
        "print(\"SHAP ANALYSIS: RANDOM FOREST\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "# TreeExplainer для Random Forest\n",
        "print(\"\\nCreating TreeExplainer for Random Forest...\")\n",
        "rf_explainer = shap.TreeExplainer(models['Random Forest'])\n",
        "\n",
        "# Вычисляем SHAP values для test set\n",
        "# Для скорости используем первые 1000 samples\n",
        "print(\"Computing SHAP values (это может занять минуту)...\")\n",
        "rf_shap_values = rf_explainer.shap_values(X_test_scaled.iloc[:1000])\n",
        "\n",
        "# SHAP возвращает [values_class_0, values_class_1] для бинарной классификации\n",
        "# Нас интересует класс 1 (>50K)\n",
        "if isinstance(rf_shap_values, list):\n",
        "    rf_shap_values_class1 = rf_shap_values[1]\n",
        "else:\n",
        "    rf_shap_values_class1 = rf_shap_values\n",
        "\n",
        "print(f\"✅ SHAP values computed\")\n",
        "print(f\"Shape: {rf_shap_values_class1.shape}\")\n",
        "print(f\"(samples={rf_shap_values_class1.shape[0]}, features={rf_shap_values_class1.shape[1]})\")\n"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "#### 2.1.1 Global Feature Importance (Summary Plot)\n",
        "\n",
        "**Summary plot** показывает:\n",
        "- Какие признаки самые важные (сверху)\n",
        "- Как значения признака влияют на prediction (цвет: red=high, blue=low)\n",
        "- Magnitude of effect (SHAP value по оси X)"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Summary plot - глобальная важность признаков\n",
        "plt.figure(figsize=(12, 8))\n",
        "shap.summary_plot(\n",
        "    rf_shap_values_class1,\n",
        "    X_test_scaled.iloc[:1000],\n",
        "    feature_names=feature_cols,\n",
        "    show=False\n",
        ")\n",
        "plt.title('SHAP Summary Plot: Random Forest (Class >50K)', fontsize=14, fontweight='bold', pad=20)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n📊 Интерпретация Summary Plot:\")\n",
        "print(\"- Features sorted by importance (top = most important)\")\n",
        "print(\"- Color: red = high feature value, blue = low feature value\")\n",
        "print(\"- X-axis: SHAP value (impact on prediction)\")\n",
        "print(\"- Positive SHAP → increases probability of >50K\")\n",
        "print(\"- Negative SHAP → decreases probability of >50K\")\n",
        "print(\"\\nПримеры:\")\n",
        "print(\"- capital-gain: high values (red) → positive SHAP → higher income probability\")\n",
        "print(\"- age: older age (red) → positive SHAP → higher income probability\")\n",
        "print(\"- education-num: more education (red) → positive SHAP → higher income probability\")\n"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Bar plot - средняя важность признаков\n",
        "plt.figure(figsize=(10, 6))\n",
        "shap.summary_plot(\n",
        "    rf_shap_values_class1,\n",
        "    X_test_scaled.iloc[:1000],\n",
        "    feature_names=feature_cols,\n",
        "    plot_type='bar',\n",
        "    show=False\n",
        ")\n",
        "plt.title('SHAP Feature Importance: Random Forest', fontsize=14, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n🎯 Топ-5 самых важных признаков:\")\n",
        "mean_abs_shap = np.abs(rf_shap_values_class1).mean(axis=0)\n",
        "feature_importance = pd.DataFrame({\n",
        "    'Feature': feature_cols,\n",
        "    'Mean |SHAP|': mean_abs_shap\n",
        "}).sort_values('Mean |SHAP|', ascending=False)\n",
        "\n",
        "print(feature_importance.head(10).to_string(index=False))\n"
    ]
})

# ============================================================================
# LOCAL EXPLANATIONS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "#### 2.1.2 Local Explanations (Waterfall Plot)\n",
        "\n",
        "**Waterfall plot** объясняет **одно конкретное prediction**:\n",
        "- Начинаем с baseline (среднее prediction по всему датасету)\n",
        "- Каждый признак двигает prediction вверх или вниз\n",
        "- Конечное значение = actual prediction"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Выберем несколько интересных samples для объяснения\n",
        "# Sample 1: High income prediction\n",
        "high_income_idx = np.where((y_test.iloc[:1000] == 1) & (results['Random Forest']['predictions'][:1000] == 1))[0][0]\n",
        "\n",
        "# Sample 2: Low income prediction\n",
        "low_income_idx = np.where((y_test.iloc[:1000] == 0) & (results['Random Forest']['predictions'][:1000] == 0))[0][0]\n",
        "\n",
        "print(f\"Selected samples:\")\n",
        "print(f\"- High income prediction: index {high_income_idx}\")\n",
        "print(f\"- Low income prediction: index {low_income_idx}\")\n",
        "\n",
        "# Waterfall plot для high income\n",
        "print(\"\\n\" + \"=\" * 70)\n",
        "print(\"EXAMPLE 1: HIGH INCOME PREDICTION\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "shap.plots.waterfall(\n",
        "    shap.Explanation(\n",
        "        values=rf_shap_values_class1[high_income_idx],\n",
        "        base_values=rf_explainer.expected_value[1] if isinstance(rf_explainer.expected_value, list) else rf_explainer.expected_value,\n",
        "        data=X_test_scaled.iloc[high_income_idx].values,\n",
        "        feature_names=feature_cols\n",
        "    ),\n",
        "    show=False\n",
        ")\n",
        "plt.title(f'Waterfall Plot: Sample {high_income_idx} (Predicted >50K)', fontsize=12, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\nАктуальные значения признаков:\")\n",
        "sample_df = pd.DataFrame({\n",
        "    'Feature': feature_cols,\n",
        "    'Value': X_test_scaled.iloc[high_income_idx].values,\n",
        "    'SHAP': rf_shap_values_class1[high_income_idx]\n",
        "}).sort_values('SHAP', key=abs, ascending=False)\n",
        "print(sample_df.head(8).to_string(index=False))\n"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Waterfall plot для low income\n",
        "print(\"\\n\" + \"=\" * 70)\n",
        "print(\"EXAMPLE 2: LOW INCOME PREDICTION\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "shap.plots.waterfall(\n",
        "    shap.Explanation(\n",
        "        values=rf_shap_values_class1[low_income_idx],\n",
        "        base_values=rf_explainer.expected_value[1] if isinstance(rf_explainer.expected_value, list) else rf_explainer.expected_value,\n",
        "        data=X_test_scaled.iloc[low_income_idx].values,\n",
        "        feature_names=feature_cols\n",
        "    ),\n",
        "    show=False\n",
        ")\n",
        "plt.title(f'Waterfall Plot: Sample {low_income_idx} (Predicted <=50K)', fontsize=12, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\nАктуальные значения признаков:\")\n",
        "sample_df = pd.DataFrame({\n",
        "    'Feature': feature_cols,\n",
        "    'Value': X_test_scaled.iloc[low_income_idx].values,\n",
        "    'SHAP': rf_shap_values_class1[low_income_idx]\n",
        "}).sort_values('SHAP', key=abs, ascending=False)\n",
        "print(sample_df.head(8).to_string(index=False))\n",
        "\n",
        "print(\"\\n💡 Ключевое преимущество SHAP:\")\n",
        "print(\"Мы можем объяснить КАЖДОЕ prediction, а не только модель в целом!\")\n",
        "print(\"Это критически важно для:\")\n",
        "print(\"- Медицинской диагностики (объяснить пациенту)\")\n",
        "print(\"- Кредитного скоринга (объяснить отказ)\")\n",
        "print(\"- Fraud detection (почему транзакция подозрительна)\")\n"
    ]
})

# ============================================================================
# DEPENDENCE PLOTS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "#### 2.1.3 Dependence Plots (Feature Interactions)\n",
        "\n",
        "**Dependence plot** показывает, как SHAP value признака зависит от его значения:\n",
        "- X-axis: значение признака\n",
        "- Y-axis: SHAP value (impact)\n",
        "- Color: взаимодействие с другим признаком (автоматически выбирается самое сильное)\n",
        "\n",
        "Помогает найти **нелинейные эффекты и interactions**."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Dependence plots для топ признаков\n",
        "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
        "axes = axes.flatten()\n",
        "\n",
        "top_features = feature_importance.head(6)['Feature'].values\n",
        "\n",
        "for idx, feature in enumerate(top_features):\n",
        "    plt.sca(axes[idx])\n",
        "    shap.dependence_plot(\n",
        "        feature,\n",
        "        rf_shap_values_class1,\n",
        "        X_test_scaled.iloc[:1000],\n",
        "        feature_names=feature_cols,\n",
        "        show=False,\n",
        "        ax=axes[idx]\n",
        "    )\n",
        "    axes[idx].set_title(f'Dependence: {feature}', fontsize=12, fontweight='bold')\n",
        "\n",
        "plt.suptitle('SHAP Dependence Plots: Feature Interactions', fontsize=14, fontweight='bold', y=1.00)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n📊 Что мы видим:\")\n",
        "print(\"- Нелинейные зависимости (не просто линейные тренды)\")\n",
        "print(\"- Feature interactions (цвет показывает взаимодействие с другим признаком)\")\n",
        "print(\"- Thresholds и breakpoints в влиянии признака\")\n",
        "print(\"\\nПример:\")\n",
        "print(\"- age: молодой возраст → negative SHAP, пожилой → positive SHAP\")\n",
        "print(\"- capital-gain: 0 → negative, >0 → strong positive SHAP\")\n"
    ]
})

# ============================================================================
# SHAP FOR XGBOOST
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.2 SHAP для XGBoost\n",
        "\n",
        "Повторим анализ для XGBoost (обычно более точная модель)."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"=\" * 70)\n",
        "print(\"SHAP ANALYSIS: XGBOOST\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "# TreeExplainer для XGBoost\n",
        "print(\"\\nCreating TreeExplainer for XGBoost...\")\n",
        "xgb_explainer = shap.TreeExplainer(models['XGBoost'])\n",
        "\n",
        "# Вычисляем SHAP values\n",
        "print(\"Computing SHAP values...\")\n",
        "xgb_shap_values = xgb_explainer.shap_values(X_test_scaled.iloc[:1000])\n",
        "\n",
        "print(f\"✅ SHAP values computed\")\n",
        "print(f\"Shape: {xgb_shap_values.shape}\")\n",
        "\n",
        "# Summary plot\n",
        "fig, axes = plt.subplots(1, 2, figsize=(20, 8))\n",
        "\n",
        "# Summary plot\n",
        "plt.sca(axes[0])\n",
        "shap.summary_plot(\n",
        "    xgb_shap_values,\n",
        "    X_test_scaled.iloc[:1000],\n",
        "    feature_names=feature_cols,\n",
        "    show=False\n",
        ")\n",
        "axes[0].set_title('SHAP Summary Plot: XGBoost', fontsize=14, fontweight='bold')\n",
        "\n",
        "# Bar plot\n",
        "plt.sca(axes[1])\n",
        "shap.summary_plot(\n",
        "    xgb_shap_values,\n",
        "    X_test_scaled.iloc[:1000],\n",
        "    feature_names=feature_cols,\n",
        "    plot_type='bar',\n",
        "    show=False\n",
        ")\n",
        "axes[1].set_title('SHAP Feature Importance: XGBoost', fontsize=14, fontweight='bold')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Feature importance comparison\n",
        "print(\"\\n🎯 Топ-10 признаков по SHAP (XGBoost):\")\n",
        "xgb_mean_abs_shap = np.abs(xgb_shap_values).mean(axis=0)\n",
        "xgb_feature_importance = pd.DataFrame({\n",
        "    'Feature': feature_cols,\n",
        "    'Mean |SHAP|': xgb_mean_abs_shap\n",
        "}).sort_values('Mean |SHAP|', ascending=False)\n",
        "\n",
        "print(xgb_feature_importance.head(10).to_string(index=False))\n"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "#### 2.2.1 Сравнение Random Forest vs XGBoost\n",
        "\n",
        "Сравним feature importance для обеих моделей."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Сравниваем feature importance RF vs XGBoost\n",
        "comparison = pd.DataFrame({\n",
        "    'Feature': feature_cols,\n",
        "    'RF SHAP': mean_abs_shap,\n",
        "    'XGB SHAP': xgb_mean_abs_shap\n",
        "}).sort_values('XGB SHAP', ascending=False)\n",
        "\n",
        "# Визуализация\n",
        "fig, ax = plt.subplots(figsize=(12, 8))\n",
        "\n",
        "x = np.arange(len(comparison.head(12)))\n",
        "width = 0.35\n",
        "\n",
        "ax.barh(x - width/2, comparison.head(12)['RF SHAP'], width, label='Random Forest', alpha=0.8)\n",
        "ax.barh(x + width/2, comparison.head(12)['XGB SHAP'], width, label='XGBoost', alpha=0.8)\n",
        "\n",
        "ax.set_yticks(x)\n",
        "ax.set_yticklabels(comparison.head(12)['Feature'])\n",
        "ax.set_xlabel('Mean |SHAP value|', fontsize=12)\n",
        "ax.set_title('Feature Importance Comparison: RF vs XGBoost', fontsize=14, fontweight='bold')\n",
        "ax.legend()\n",
        "ax.invert_yaxis()\n",
        "ax.grid(axis='x', alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n🔍 Наблюдения:\")\n",
        "print(\"- Обе модели согласны в топовых признаках (capital-gain, age, education-num)\")\n",
        "print(\"- Ranking может немного отличаться\")\n",
        "print(\"- XGBoost может выявлять более тонкие interactions\")\n",
        "print(\"\\n→ Консенсус между моделями повышает уверенность в интерпретации!\")\n"
    ]
})

# Сохраняем обновленный notebook
notebook['cells'] = cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'\\n✅ Updated notebook: {notebook_path}')
print(f'Total cells: {len(cells)}')
print('SHAP analysis added!')
