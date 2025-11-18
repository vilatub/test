#!/usr/bin/env python3
"""
Phase 4 Step 2: TabTransformer for Tabular Data
Part 1: Introduction, Adult Income Dataset, EDA
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
        "# 📊 TabTransformer: Transformers для Табличных Данных\n",
        "\n",
        "**Phase 4, Step 2: Advanced Transformer Architectures**\n",
        "\n",
        "---\n",
        "\n",
        "## 🎯 Проблема Categorical Features\n",
        "\n",
        "### В Phase 4 Step 1 мы видели:\n",
        "\n",
        "**Обычный Transformer для табличных данных:**\n",
        "- ✅ Feature embedding: Linear projection\n",
        "- ✅ Self-Attention между features\n",
        "- ✅ Работает на Titanic (891 samples)\n",
        "\n",
        "**Но были проблемы:**\n",
        "- ❌ **Маленький датасет**: Titanic слишком мал для Transformers\n",
        "- ❌ **Categorical features**: просто one-hot encoded\n",
        "- ❌ **No contextual embeddings**: категории не учат друг у друга\n",
        "- ❌ **Не показывает преимущества**: XGBoost работал так же хорошо\n",
        "\n",
        "---\n",
        "\n",
        "## 🚀 Enter TabTransformer (2020)\n",
        "\n",
        "**\"TabTransformer: Tabular Data Modeling Using Contextual Embeddings\"** (Huang et al., 2020)\n",
        "\n",
        "**Ключевая идея:** Categorical features → **Contextual Embeddings** через Transformer!\n",
        "\n",
        "### Архитектура TabTransformer:\n",
        "\n",
        "```\n",
        "Input: [Cat1, Cat2, Cat3, ..., CatM] + [Num1, Num2, ..., NumN]\n",
        "          ↓         ↓       ↓\n",
        "    [Emb1]   [Emb2]   [Emb3]  ← Column Embeddings (learnable)\n",
        "          ↓         ↓       ↓\n",
        "      + Positional Encoding\n",
        "          ↓         ↓       ↓\n",
        "    ┌─────────────────────────┐\n",
        "    │  Transformer Layers     │  ← Attention между categorical features\n",
        "    │  (N encoder blocks)     │\n",
        "    └─────────────────────────┘\n",
        "          ↓         ↓       ↓\n",
        "    [Ctx1]   [Ctx2]   [Ctx3]  ← Contextual Embeddings\n",
        "          └─────────┴─────────┘\n",
        "                  ↓\n",
        "          Concatenate with [Num1, Num2, ..., NumN]\n",
        "                  ↓\n",
        "            MLP Classifier\n",
        "                  ↓\n",
        "              Output\n",
        "```\n",
        "\n",
        "### Ключевые отличия от обычного Transformer:\n",
        "\n",
        "1. **Column Embeddings** вместо Linear Projection:\n",
        "   - Каждая categorical feature → lookup embedding (как word embeddings)\n",
        "   - Размерность: `vocab_size × d_model`\n",
        "   - Аналог word2vec для категорий\n",
        "\n",
        "2. **Transformer только на Categorical**:\n",
        "   - Transformer обрабатывает только categorical features\n",
        "   - Numerical features остаются как есть\n",
        "   - Concatenation в конце\n",
        "\n",
        "3. **Contextual Embeddings**:\n",
        "   - После Transformer каждая категория имеет контекст других категорий\n",
        "   - Пример: \"Occupation=Teacher\" + \"Education=Masters\" → контекстное представление\n",
        "\n",
        "---\n",
        "\n",
        "## 📊 Adult Income Dataset\n",
        "\n",
        "**Задача:** Предсказать, зарабатывает ли человек >$50K/год\n",
        "\n",
        "**Размер:** ~48,842 samples (в 55 раз больше Titanic!)\n",
        "\n",
        "**Features (14 total):**\n",
        "\n",
        "**Categorical (8):**\n",
        "- `workclass`: Private, Self-emp, Federal-gov, etc. (9 categories)\n",
        "- `education`: Bachelors, HS-grad, Masters, Doctorate, etc. (16 categories)\n",
        "- `marital-status`: Married, Never-married, Divorced, etc. (7 categories)\n",
        "- `occupation`: Tech-support, Craft-repair, Sales, Exec-managerial, etc. (15 categories)\n",
        "- `relationship`: Wife, Husband, Not-in-family, etc. (6 categories)\n",
        "- `race`: White, Black, Asian-Pac-Islander, etc. (5 categories)\n",
        "- `sex`: Male, Female (2 categories)\n",
        "- `native-country`: United-States, Mexico, India, etc. (42 categories)\n",
        "\n",
        "**Numerical (6):**\n",
        "- `age`: Возраст\n",
        "- `fnlwgt`: Final weight (census weight)\n",
        "- `education-num`: Годы образования\n",
        "- `capital-gain`: Capital gain\n",
        "- `capital-loss`: Capital loss\n",
        "- `hours-per-week`: Часов работы в неделю\n",
        "\n",
        "**Target:** `income` (>50K или <=50K)\n",
        "\n",
        "**Почему идеален для TabTransformer:**\n",
        "- ✅ Большой датасет (>40k samples)\n",
        "- ✅ Много categorical features (8 штук)\n",
        "- ✅ Высокая cardinality (education=16, occupation=15, country=42)\n",
        "- ✅ Сложные взаимодействия (education × occupation × marital-status)\n",
        "- ✅ Классический benchmark для табличных данных\n",
        "\n",
        "---\n",
        "\n",
        "## 🎯 Что мы покажем\n",
        "\n",
        "1. **Contextual Embeddings работают:**\n",
        "   - TabTransformer > обычный Transformer\n",
        "   - Categorical embeddings учатся в контексте друг друга\n",
        "\n",
        "2. **Competitive с Tree-based:**\n",
        "   - TabTransformer ≈ XGBoost/LightGBM на большом датасете\n",
        "   - Преимущество: интерпретируемость через attention\n",
        "\n",
        "3. **Attention показывает interactions:**\n",
        "   - Какие categorical features взаимодействуют\n",
        "   - \"Education\" attention на \"Occupation\"\n",
        "\n",
        "4. **Масштабируемость:**\n",
        "   - 48k samples → показывает силу Deep Learning\n",
        "   - Не как Titanic (891 samples)\n",
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
        "## 💻 Часть 1: Подготовка данных\n",
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
        "from collections import Counter\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# PyTorch\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import torch.nn.functional as F\n",
        "from torch.utils.data import Dataset, DataLoader, TensorDataset\n",
        "\n",
        "# Sklearn\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
        "from sklearn.metrics import (\n",
        "    accuracy_score, precision_score, recall_score, f1_score,\n",
        "    classification_report, confusion_matrix, roc_auc_score, roc_curve\n",
        ")\n",
        "\n",
        "# Math\n",
        "import math\n",
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
        "torch.manual_seed(42)\n",
        "np.random.seed(42)\n",
        "\n",
        "print(\"\\n✅ Все библиотеки загружены\")"
    ]
})

# ============================================================================
# DATA LOADING
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.2 Загрузка Adult Income Dataset\n",
        "\n",
        "**Источник:** UCI Machine Learning Repository  \n",
        "**URL:** https://archive.ics.uci.edu/ml/datasets/adult\n"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Column names\n",
        "column_names = [\n",
        "    'age', 'workclass', 'fnlwgt', 'education', 'education-num',\n",
        "    'marital-status', 'occupation', 'relationship', 'race', 'sex',\n",
        "    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'\n",
        "]\n",
        "\n",
        "# URLs\n",
        "train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'\n",
        "test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'\n",
        "\n",
        "print(\"Загрузка Adult Income Dataset...\")\n",
        "\n",
        "try:\n",
        "    # Загружаем train и test\n",
        "    df_train = pd.read_csv(train_url, names=column_names, na_values=' ?', skipinitialspace=True)\n",
        "    df_test = pd.read_csv(test_url, names=column_names, na_values=' ?', skipinitialspace=True, skiprows=1)\n",
        "    \n",
        "    # Объединяем\n",
        "    df = pd.concat([df_train, df_test], ignore_index=True)\n",
        "    \n",
        "    print(f\"✅ Загружено {len(df)} samples\")\n",
        "    print(f\"   Train: {len(df_train)} samples\")\n",
        "    print(f\"   Test: {len(df_test)} samples\")\n",
        "    \n",
        "except Exception as e:\n",
        "    print(f\"❌ Ошибка загрузки: {e}\")\n",
        "    print(\"\\nСоздаем синтетические данные для демонстрации...\")\n",
        "    \n",
        "    # Синтетические данные\n",
        "    np.random.seed(42)\n",
        "    n_samples = 48842\n",
        "    \n",
        "    df = pd.DataFrame({\n",
        "        'age': np.random.randint(17, 90, n_samples),\n",
        "        'workclass': np.random.choice(['Private', 'Self-emp', 'Federal-gov', 'Local-gov', 'State-gov'], n_samples),\n",
        "        'fnlwgt': np.random.randint(10000, 500000, n_samples),\n",
        "        'education': np.random.choice(['Bachelors', 'HS-grad', 'Masters', 'Some-college', 'Assoc', 'Doctorate'], n_samples),\n",
        "        'education-num': np.random.randint(1, 16, n_samples),\n",
        "        'marital-status': np.random.choice(['Married', 'Never-married', 'Divorced', 'Separated', 'Widowed'], n_samples),\n",
        "        'occupation': np.random.choice(['Tech-support', 'Craft-repair', 'Sales', 'Exec-managerial', 'Prof-specialty'], n_samples),\n",
        "        'relationship': np.random.choice(['Husband', 'Wife', 'Not-in-family', 'Own-child', 'Unmarried'], n_samples),\n",
        "        'race': np.random.choice(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], n_samples),\n",
        "        'sex': np.random.choice(['Male', 'Female'], n_samples),\n",
        "        'capital-gain': np.random.choice([0] * 90 + list(range(1000, 100000, 1000)), n_samples),\n",
        "        'capital-loss': np.random.choice([0] * 90 + list(range(1000, 5000, 100)), n_samples),\n",
        "        'hours-per-week': np.random.randint(1, 99, n_samples),\n",
        "        'native-country': np.random.choice(['United-States', 'Mexico', 'India', 'Philippines', 'Germany'], n_samples),\n",
        "    })\n",
        "    \n",
        "    # Target: синтетическая логика\n",
        "    income_prob = (\n",
        "        (df['age'] > 30).astype(int) * 0.2 +\n",
        "        (df['education-num'] > 12).astype(int) * 0.3 +\n",
        "        (df['hours-per-week'] > 40).astype(int) * 0.2 +\n",
        "        (df['capital-gain'] > 0).astype(int) * 0.3\n",
        "    )\n",
        "    df['income'] = (np.random.random(n_samples) < income_prob).astype(int)\n",
        "    df['income'] = df['income'].map({0: '<=50K', 1: '>50K'})\n",
        "\n",
        "print(f\"\\nDataset shape: {df.shape}\")\n",
        "print(f\"Columns: {df.columns.tolist()}\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Первые строки\n",
        "print(\"Первые 5 строк:\")\n",
        "df.head()"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Базовая информация\n",
        "print(\"Информация о датасете:\")\n",
        "print(df.info())\n",
        "\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"СТАТИСТИКА\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "# Numerical features\n",
        "numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n",
        "print(f\"\\nNumerical features ({len(numerical_cols)}): {numerical_cols}\")\n",
        "print(df[numerical_cols].describe())\n",
        "\n",
        "# Categorical features\n",
        "categorical_cols = df.select_dtypes(include=['object']).columns.tolist()\n",
        "categorical_cols.remove('income')  # убираем target\n",
        "print(f\"\\nCategorical features ({len(categorical_cols)}): {categorical_cols}\")\n",
        "\n",
        "# Missing values\n",
        "print(\"\\nПропуски:\")\n",
        "missing = df.isnull().sum()\n",
        "if missing.sum() > 0:\n",
        "    print(missing[missing > 0])\n",
        "    print(f\"\\nВсего пропусков: {missing.sum()} ({missing.sum() / len(df) * 100:.2f}%)\")\n",
        "else:\n",
        "    print(\"Нет пропусков ✅\")"
    ]
})

# ============================================================================
# EDA
# ============================================================================

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
        "# Target distribution\n",
        "print(\"Target Distribution (Income):\")\n",
        "print(df['income'].value_counts())\n",
        "print(f\"\\n>50K rate: {(df['income'] == '>50K').mean():.2%}\")\n",
        "\n",
        "# Visualize\n",
        "fig, ax = plt.subplots(1, 1, figsize=(8, 6))\n",
        "df['income'].value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'orange'])\n",
        "ax.set_title('Income Distribution', fontsize=16, fontweight='bold')\n",
        "ax.set_xlabel('Income', fontsize=12)\n",
        "ax.set_ylabel('Count', fontsize=12)\n",
        "ax.set_xticklabels(ax.get_xticklabels(), rotation=0)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n📊 Dataset slightly imbalanced but acceptable for classification\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Categorical features cardinality\n",
        "print(\"Categorical Features Cardinality:\")\n",
        "print(\"=\"*50)\n",
        "\n",
        "cardinality = {}\n",
        "for col in categorical_cols:\n",
        "    n_unique = df[col].nunique()\n",
        "    cardinality[col] = n_unique\n",
        "    print(f\"{col:20s}: {n_unique:3d} unique values\")\n",
        "\n",
        "print(\"\\n📊 High cardinality в native-country (42), education (16), occupation (15)\")\n",
        "print(\"   Это идеально для contextual embeddings!\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualize categorical features\n",
        "fig, axes = plt.subplots(3, 3, figsize=(18, 12))\n",
        "axes = axes.flatten()\n",
        "\n",
        "for idx, col in enumerate(categorical_cols):\n",
        "    if idx >= 9:\n",
        "        break\n",
        "    \n",
        "    # Count by income\n",
        "    pd.crosstab(df[col], df['income']).plot(kind='bar', ax=axes[idx], \n",
        "                                             color=['steelblue', 'orange'])\n",
        "    axes[idx].set_title(f'{col}', fontsize=12, fontweight='bold')\n",
        "    axes[idx].set_xlabel('')\n",
        "    axes[idx].legend(['<=50K', '>50K'], loc='upper right')\n",
        "    axes[idx].tick_params(axis='x', labelsize=8, rotation=45)\n",
        "\n",
        "plt.suptitle('Categorical Features Distribution by Income', \n",
        "             fontsize=16, fontweight='bold', y=1.00)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Numerical features distribution\n",
        "fig, axes = plt.subplots(2, 3, figsize=(16, 10))\n",
        "axes = axes.flatten()\n",
        "\n",
        "for idx, col in enumerate(numerical_cols):\n",
        "    df[df['income'] == '<=50K'][col].hist(bins=30, alpha=0.5, label='<=50K', \n",
        "                                          ax=axes[idx], color='steelblue')\n",
        "    df[df['income'] == '>50K'][col].hist(bins=30, alpha=0.5, label='>50K', \n",
        "                                         ax=axes[idx], color='orange')\n",
        "    axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')\n",
        "    axes[idx].set_xlabel(col)\n",
        "    axes[idx].legend()\n",
        "    axes[idx].grid(alpha=0.3)\n",
        "\n",
        "plt.suptitle('Numerical Features Distribution by Income', \n",
        "             fontsize=16, fontweight='bold', y=1.00)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n📊 Key observations:\")\n",
        "print(\"  - Age: higher income for 35-55 age group\")\n",
        "print(\"  - Education-num: clear correlation with income\")\n",
        "print(\"  - Capital-gain/loss: strong predictors (but sparse)\")\n",
        "print(\"  - Hours-per-week: >50K work slightly more hours\")"
    ]
})

# Сохраняем первую часть
notebook['cells'] = cells

output_path = '/home/user/test/notebooks/phase4_transformers/02_tabtransformer.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Part 1 создана: {output_path}')
print(f'Ячеек: {len(cells)}')
print('Следующая часть: TabTransformer Theory and Implementation...')
