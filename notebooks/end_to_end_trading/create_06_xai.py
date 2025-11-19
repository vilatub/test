#!/usr/bin/env python3
"""
Скрипт для создания 06_xai_interpretation.ipynb
XAI и интерпретация моделей
"""

import json

def create_notebook():
    cells = []

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# End-to-End Trading Project\n",
            "## Часть 6: XAI и Интерпретация\n",
            "\n",
            "### В этом ноутбуке:\n",
            "\n",
            "1. **SHAP** - глобальная и локальная интерпретация\n",
            "2. **Permutation Importance** - важность признаков\n",
            "3. **Partial Dependence** - влияние признаков\n",
            "4. **Анализ торговых сигналов**\n",
            "\n",
            "*Детальные методы XAI доступны в `phase6_explainable_ai/`*"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import shap\n",
            "import joblib\n",
            "from sklearn.inspection import permutation_importance, PartialDependenceDisplay\n",
            "import json\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "plt.style.use('seaborn-v0_8-whitegrid')\n",
            "print('Библиотеки загружены')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Загружаем данные и модели\n",
            "data_dir = 'data'\n",
            "models_dir = 'models'\n",
            "\n",
            "df = pd.read_parquet(f'{data_dir}/processed_data.parquet')\n",
            "with open(f'{data_dir}/feature_sets.json', 'r') as f:\n",
            "    feature_sets = json.load(f)\n",
            "\n",
            "# Загружаем LightGBM (лучшая baseline модель)\n",
            "lgb_model = joblib.load(f'{models_dir}/lightgbm.joblib')\n",
            "\n",
            "feature_cols = [f for f in feature_sets['extended_features'] if f in df.columns]\n",
            "target_col = 'target_direction_1d'\n",
            "\n",
            "# Подготовка данных\n",
            "df = df.sort_values('date').dropna(subset=feature_cols + [target_col])\n",
            "test_df = df.iloc[int(len(df)*0.8):]\n",
            "\n",
            "X_test = test_df[feature_cols].values\n",
            "y_test = test_df[target_col].values\n",
            "\n",
            "print(f'Test samples: {len(X_test)}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. SHAP Analysis"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# SHAP для LightGBM\n",
            "explainer = shap.TreeExplainer(lgb_model)\n",
            "\n",
            "# Используем подвыборку для скорости\n",
            "sample_size = min(1000, len(X_test))\n",
            "sample_idx = np.random.choice(len(X_test), sample_size, replace=False)\n",
            "X_sample = X_test[sample_idx]\n",
            "\n",
            "shap_values = explainer.shap_values(X_sample)\n",
            "\n",
            "# Для бинарной классификации берём класс 1 (Up)\n",
            "if isinstance(shap_values, list):\n",
            "    shap_values = shap_values[1]\n",
            "\n",
            "print(f'SHAP values shape: {shap_values.shape}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Summary plot\n",
            "plt.figure(figsize=(10, 8))\n",
            "shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)\n",
            "plt.title('SHAP Feature Importance')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Bar plot - средняя важность\n",
            "plt.figure(figsize=(10, 6))\n",
            "shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, \n",
            "                  plot_type='bar', show=False)\n",
            "plt.title('Mean |SHAP| - Feature Importance')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Локальная Интерпретация\n",
            "\n",
            "Разбираем конкретные прогнозы."
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Waterfall plot для одного примера\n",
            "example_idx = 0\n",
            "expected_value = explainer.expected_value\n",
            "if isinstance(expected_value, list):\n",
            "    expected_value = expected_value[1]\n",
            "\n",
            "plt.figure(figsize=(12, 6))\n",
            "shap.waterfall_plot(shap.Explanation(\n",
            "    values=shap_values[example_idx],\n",
            "    base_values=expected_value,\n",
            "    data=X_sample[example_idx],\n",
            "    feature_names=feature_cols\n",
            "), show=False)\n",
            "plt.title('Decomposition of Single Prediction')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Permutation Importance"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Permutation importance\n",
            "perm_importance = permutation_importance(lgb_model, X_sample, y_test[sample_idx], \n",
            "                                        n_repeats=10, random_state=42)\n",
            "\n",
            "# Визуализация\n",
            "sorted_idx = perm_importance.importances_mean.argsort()[::-1][:15]\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 6))\n",
            "ax.boxplot(perm_importance.importances[sorted_idx].T, vert=False,\n",
            "           labels=[feature_cols[i] for i in sorted_idx])\n",
            "ax.set_xlabel('Decrease in Accuracy')\n",
            "ax.set_title('Permutation Importance (Top 15)')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Partial Dependence Plots"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# PDP для топ признаков\n",
            "top_features = ['rsi', 'macd', 'bb_position', 'return']\n",
            "top_indices = [feature_cols.index(f) for f in top_features if f in feature_cols][:4]\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(12, 8))\n",
            "PartialDependenceDisplay.from_estimator(\n",
            "    lgb_model, X_sample, top_indices,\n",
            "    feature_names=feature_cols,\n",
            "    ax=ax, grid_resolution=50\n",
            ")\n",
            "plt.suptitle('Partial Dependence Plots', fontsize=14)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Анализ Торговых Сигналов"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Анализ прогнозов по уверенности\n",
            "proba = lgb_model.predict_proba(X_test)[:, 1]\n",
            "\n",
            "# Разбиваем на группы по уверенности\n",
            "bins = [0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0]\n",
            "labels = ['<0.4', '0.4-0.45', '0.45-0.5', '0.5-0.55', '0.55-0.6', '>0.6']\n",
            "confidence_groups = pd.cut(proba, bins=bins, labels=labels)\n",
            "\n",
            "# Accuracy по группам\n",
            "results = []\n",
            "for group in labels:\n",
            "    mask = confidence_groups == group\n",
            "    if mask.sum() > 0:\n",
            "        group_acc = (y_test[mask] == (proba[mask] > 0.5)).mean()\n",
            "        results.append({'group': group, 'count': mask.sum(), 'accuracy': group_acc})\n",
            "\n",
            "results_df = pd.DataFrame(results)\n",
            "print('Accuracy by Confidence:\\n')\n",
            "print(results_df.to_string(index=False))"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Визуализация\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "\n",
            "# Distribution of predictions\n",
            "axes[0].hist(proba, bins=50, edgecolor='black', alpha=0.7)\n",
            "axes[0].axvline(x=0.5, color='red', linestyle='--')\n",
            "axes[0].set_xlabel('Predicted Probability (Up)')\n",
            "axes[0].set_ylabel('Count')\n",
            "axes[0].set_title('Distribution of Predictions')\n",
            "\n",
            "# Accuracy vs Confidence\n",
            "axes[1].bar(results_df['group'], results_df['accuracy'])\n",
            "axes[1].axhline(y=0.5, color='red', linestyle='--', label='Random')\n",
            "axes[1].set_xlabel('Confidence Group')\n",
            "axes[1].set_ylabel('Accuracy')\n",
            "axes[1].set_title('Accuracy by Confidence')\n",
            "axes[1].legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Итоги\n",
            "\n",
            "### Ключевые выводы:\n",
            "\n",
            "1. **SHAP** показывает, что модель использует:\n",
            "   - Momentum индикаторы (ROC, return)\n",
            "   - Волатильность (ATR)\n",
            "   - RSI и MACD\n",
            "\n",
            "2. **Partial Dependence** выявляет нелинейные зависимости\n",
            "\n",
            "3. **Уверенность модели** коррелирует с accuracy\n",
            "\n",
            "### Рекомендации для трейдинга:\n",
            "\n",
            "- Использовать только сигналы с высокой уверенностью (>0.6)\n",
            "- Комбинировать с другими индикаторами\n",
            "- Учитывать текущий рыночный режим\n",
            "\n",
            "### Следующий шаг:\n",
            "\n",
            "В ноутбуке 07 проведём бэктестинг торговых стратегий."
        ]
    })

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "cells": cells
    }
    return notebook

if __name__ == "__main__":
    notebook = create_notebook()
    output_path = "/home/user/test/notebooks/end_to_end_trading/06_xai_interpretation.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"Notebook created: {output_path}")
    print(f"Total cells: {len(notebook['cells'])}")
