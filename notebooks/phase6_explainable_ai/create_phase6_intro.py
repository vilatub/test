#!/usr/bin/env python3
"""
Phase 6: Explainable AI (XAI) - Interpretability & Fairness
Part 1: Introduction, Setup, Dataset, Model Training
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
        "# 🔬 Explainable AI (XAI): Interpretability & Fairness\n",
        "\n",
        "**Phase 6: Understanding HOW and WHY ML Models Make Decisions**\n",
        "\n",
        "---\n",
        "\n",
        "## 🎯 The Black Box Problem\n",
        "\n",
        "### До сих пор мы фокусировались на точности:\n",
        "\n",
        "- ✅ **Phase 1-2:** Accuracy, AUC, F1-Score\n",
        "- ✅ **Phase 3:** Ensemble methods для улучшения метрик\n",
        "- ✅ **Phase 4:** Transformers для комплексных паттернов\n",
        "- ✅ **Phase 5:** Unsupervised learning для anomaly detection\n",
        "\n",
        "**Но в реальном мире этого недостаточно:**\n",
        "\n",
        "### 🏥 Медицина\n",
        "```\n",
        "Модель: \"У пациента рак легких с вероятностью 85%\"\n",
        "Врач: \"Почему? На основании каких признаков?\"\n",
        "Модель: \"🤷 (black box)\"\n",
        "```\n",
        "❌ **НЕПРИЕМЛЕМО** - FDA и GDPR требуют объяснимости\n",
        "\n",
        "### 💳 Финансы\n",
        "```\n",
        "Модель: \"Кредит отклонен\"\n",
        "Клиент: \"Почему? Что нужно улучшить?\"\n",
        "Банк: \"🤷 (black box)\"\n",
        "```\n",
        "❌ **ILLEGAL** - законодательство требует объяснений (Equal Credit Opportunity Act)\n",
        "\n",
        "### 🏢 Рекрутинг\n",
        "```\n",
        "Модель: 95% мужчин получают job offers для tech позиций\n",
        "HR: \"Модель дискриминирует по полу?\"\n",
        "Data Scientist: \"🤷 Accuracy 92%, что не так?\"\n",
        "```\n",
        "❌ **BIAS PROBLEM** - высокая точность != fair predictions\n",
        "\n",
        "---\n",
        "\n",
        "## 🚀 Enter Explainable AI (XAI)\n",
        "\n",
        "### Ключевые вопросы:\n",
        "\n",
        "1. **Global Interpretability:** Как модель работает в целом?\n",
        "   - Какие признаки самые важные?\n",
        "   - Как признаки влияют на predictions?\n",
        "   - Есть ли нелинейные взаимодействия?\n",
        "\n",
        "2. **Local Interpretability:** Почему именно это prediction?\n",
        "   - Почему этому клиенту отказали в кредите?\n",
        "   - Какие факторы повлияли на этот диагноз?\n",
        "   - Что нужно изменить для другого результата?\n",
        "\n",
        "3. **Fairness:** Модель справедлива?\n",
        "   - Есть ли bias по полу, расе, возрасту?\n",
        "   - Предсказания калиброваны для всех групп?\n",
        "   - Demographic parity vs Equal opportunity?\n",
        "\n",
        "---\n",
        "\n",
        "## 📊 XAI Methods Overview\n",
        "\n",
        "### 1. Model-Agnostic Methods (работают с любой моделью)\n",
        "\n",
        "#### **SHAP (SHapley Additive exPlanations)**\n",
        "- ✅ **Теоретически обоснован** (game theory, Shapley values)\n",
        "- ✅ **Consistent:** если модель больше полагается на признак, SHAP value выше\n",
        "- ✅ **Local + Global interpretability**\n",
        "- ⚠️ **Computational cost:** TreeSHAP быстр, KernelSHAP медленнее\n",
        "\n",
        "**Variants:**\n",
        "- `TreeSHAP`: для tree-based моделей (XGBoost, RandomForest) - очень быстро\n",
        "- `KernelSHAP`: для любых моделей - медленнее\n",
        "- `DeepSHAP`: для нейронных сетей\n",
        "\n",
        "#### **LIME (Local Interpretable Model-agnostic Explanations)**\n",
        "- ✅ **Fast:** быстрее SHAP для локальных объяснений\n",
        "- ✅ **Intuitive:** аппроксимирует модель локально простой моделью\n",
        "- ⚠️ **Unstable:** результаты могут варьироваться\n",
        "- ⚠️ **Only local:** не даёт глобальной картины\n",
        "\n",
        "#### **Partial Dependence Plots (PDP)**\n",
        "- ✅ **Global view:** влияние признака на predictions в среднем\n",
        "- ✅ **Easy to interpret:** визуально понятно\n",
        "- ⚠️ **Assumes independence:** может вводить в заблуждение при корреляциях\n",
        "\n",
        "#### **Permutation Importance**\n",
        "- ✅ **Simple:** shuffle признак → measure drop in accuracy\n",
        "- ✅ **True importance:** учитывает корреляции (в отличие от Gini importance)\n",
        "- ⚠️ **Computational cost:** требует многих перестановок\n",
        "\n",
        "---\n",
        "\n",
        "### 2. Model-Specific Methods\n",
        "\n",
        "#### **Feature Importance (Tree-based models)**\n",
        "- ✅ **Built-in:** быстро, доступно из коробки\n",
        "- ⚠️ **Gini bias:** переоценивает high-cardinality features\n",
        "\n",
        "#### **Attention Weights (Transformers)**\n",
        "- ✅ **Direct:** модель \"говорит\", на что смотрит\n",
        "- ⚠️ **Interpretation caveats:** attention ≠ importance (спорный момент)\n",
        "\n",
        "#### **Linear Model Coefficients**\n",
        "- ✅ **Direct interpretation:** вес = влияние признака\n",
        "- ⚠️ **Only linear models:** не работает для deep learning\n",
        "\n",
        "---\n",
        "\n",
        "## 📊 Что мы реализуем\n",
        "\n",
        "### Dataset: Income Prediction (Adult Census)\n",
        "\n",
        "**Почему этот датасет?**\n",
        "- ✅ **Fairness concerns:** пол, раса, возраст могут создавать bias\n",
        "- ✅ **Real-world problem:** income prediction важен для кредитного скоринга\n",
        "- ✅ **Interpretability важна:** объяснение, почему кто-то в high/low income группе\n",
        "- ✅ **Multiple feature types:** numerical, categorical\n",
        "\n",
        "### Задачи:\n",
        "\n",
        "**Part 1: Setup & Model Training**\n",
        "1. Загрузка Adult Census dataset\n",
        "2. Preprocessing\n",
        "3. Обучение нескольких моделей (Logistic Regression, RandomForest, XGBoost)\n",
        "\n",
        "**Part 2: SHAP Analysis**\n",
        "1. TreeSHAP для RandomForest и XGBoost\n",
        "2. Global feature importance (summary plots)\n",
        "3. Local explanations (waterfall plots, force plots)\n",
        "4. Dependence plots (feature interactions)\n",
        "\n",
        "**Part 3: LIME Analysis**\n",
        "1. Local explanations для отдельных predictions\n",
        "2. Сравнение SHAP vs LIME\n",
        "\n",
        "**Part 4: Global Interpretability**\n",
        "1. Partial Dependence Plots (PDP)\n",
        "2. Individual Conditional Expectation (ICE) curves\n",
        "3. Permutation Importance\n",
        "\n",
        "**Part 5: Fairness Analysis**\n",
        "1. Demographic Parity по полу\n",
        "2. Equal Opportunity analysis\n",
        "3. Calibration по группам\n",
        "4. Bias mitigation strategies\n",
        "\n",
        "**Part 6: Decision Tree Visualization**\n",
        "1. Визуализация правил Decision Tree\n",
        "2. Rule extraction\n",
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
        "# Sklearn - Models\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "import xgboost as xgb\n",
        "\n",
        "# Sklearn - Metrics\n",
        "from sklearn.metrics import (\n",
        "    accuracy_score, precision_score, recall_score, f1_score,\n",
        "    roc_auc_score, confusion_matrix, classification_report,\n",
        "    roc_curve, precision_recall_curve\n",
        ")\n",
        "\n",
        "# Sklearn - Interpretability\n",
        "from sklearn.inspection import (\n",
        "    permutation_importance,\n",
        "    PartialDependenceDisplay,\n",
        "    partial_dependence\n",
        ")\n",
        "from sklearn.tree import plot_tree, export_text\n",
        "\n",
        "# SHAP\n",
        "import shap\n",
        "shap.initjs()  # для визуализации в Jupyter\n",
        "\n",
        "# LIME\n",
        "import lime\n",
        "import lime.lime_tabular\n",
        "\n",
        "# Настройки визуализации\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "sns.set_palette(\"husl\")\n",
        "%matplotlib inline\n",
        "\n",
        "# Reproducibility\n",
        "np.random.seed(42)\n",
        "\n",
        "print(\"\\n✅ Все библиотеки загружены\")\n",
        "print(f\"SHAP version: {shap.__version__}\")\n",
        "print(f\"LIME version: {lime.__version__}\")\n"
    ]
})

# ============================================================================
# DATASET LOADING
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.2 Загрузка Adult Census Dataset\n",
        "\n",
        "**Adult Income Dataset** (также известен как Census Income):\n",
        "- **Задача:** Предсказать, зарабатывает ли человек >50K в год\n",
        "- **Размер:** ~48,000 записей\n",
        "- **Признаки:** age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country\n",
        "- **Target:** income (<=50K or >50K)\n",
        "\n",
        "**Важно для XAI:**\n",
        "- Sensitive attributes: sex, race → fairness analysis\n",
        "- Categorical features → интерпретация категорий\n",
        "- Real-world implications → ethical AI"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Загружаем Adult Census dataset\n",
        "# Используем встроенный dataset или загружаем из UCI repository\n",
        "\n",
        "url = \"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data\"\n",
        "\n",
        "column_names = [\n",
        "    'age', 'workclass', 'fnlwgt', 'education', 'education-num',\n",
        "    'marital-status', 'occupation', 'relationship', 'race', 'sex',\n",
        "    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'\n",
        "]\n",
        "\n",
        "# Загружаем данные\n",
        "try:\n",
        "    df = pd.read_csv(url, names=column_names, sep=',\\s*', engine='python', na_values='?')\n",
        "    print(\"✅ Dataset загружен из UCI repository\")\n",
        "except:\n",
        "    print(\"⚠️ Не удалось загрузить из UCI, создаю синтетический dataset\")\n",
        "    # Создаём синтетический dataset, если нет интернета\n",
        "    np.random.seed(42)\n",
        "    n_samples = 30000\n",
        "    \n",
        "    df = pd.DataFrame({\n",
        "        'age': np.random.randint(17, 90, n_samples),\n",
        "        'workclass': np.random.choice(['Private', 'Self-emp', 'Govt', 'Without-pay'], n_samples, p=[0.7, 0.15, 0.1, 0.05]),\n",
        "        'education': np.random.choice(['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Doctorate'], n_samples, p=[0.3, 0.25, 0.3, 0.1, 0.05]),\n",
        "        'education-num': np.random.randint(1, 16, n_samples),\n",
        "        'marital-status': np.random.choice(['Married', 'Never-married', 'Divorced'], n_samples, p=[0.5, 0.35, 0.15]),\n",
        "        'occupation': np.random.choice(['Tech', 'Sales', 'Service', 'Craft', 'Prof'], n_samples, p=[0.15, 0.25, 0.25, 0.2, 0.15]),\n",
        "        'relationship': np.random.choice(['Husband', 'Wife', 'Own-child', 'Not-in-family'], n_samples, p=[0.3, 0.25, 0.2, 0.25]),\n",
        "        'race': np.random.choice(['White', 'Black', 'Asian', 'Other'], n_samples, p=[0.8, 0.1, 0.05, 0.05]),\n",
        "        'sex': np.random.choice(['Male', 'Female'], n_samples, p=[0.67, 0.33]),\n",
        "        'capital-gain': np.random.choice([0] * 90 + list(np.random.randint(1000, 100000, 10)), n_samples),\n",
        "        'capital-loss': np.random.choice([0] * 95 + list(np.random.randint(100, 5000, 5)), n_samples),\n",
        "        'hours-per-week': np.random.randint(1, 100, n_samples),\n",
        "        'native-country': np.random.choice(['United-States', 'Other'], n_samples, p=[0.9, 0.1]),\n",
        "    })\n",
        "    \n",
        "    # Создаём target с некоторой логикой\n",
        "    income_prob = (\n",
        "        (df['age'] > 30).astype(int) * 0.2 +\n",
        "        (df['education-num'] > 12).astype(int) * 0.3 +\n",
        "        (df['hours-per-week'] > 40).astype(int) * 0.2 +\n",
        "        (df['capital-gain'] > 0).astype(int) * 0.25\n",
        "    ) / 1.0\n",
        "    \n",
        "    df['income'] = (np.random.random(n_samples) < income_prob).astype(int)\n",
        "    df['income'] = df['income'].map({0: '<=50K', 1: '>50K'})\n",
        "\n",
        "# Убираем пробелы из категориальных признаков\n",
        "df = df.apply(lambda x: x.str.strip() if x.dtype == \"object\" else x)\n",
        "\n",
        "# Удаляем строки с пропусками\n",
        "df = df.dropna()\n",
        "\n",
        "print(f\"\\nРазмер датасета: {df.shape}\")\n",
        "print(f\"Признаков: {df.shape[1] - 1}\")\n",
        "print(f\"\\nПервые строки:\")\n",
        "df.head()\n"
    ]
})

# Сохраняем промежуточный результат
notebook['cells'] = cells

output_path = '/home/user/test/notebooks/phase6_explainable_ai/01_explainable_ai_xai.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Started notebook: {output_path}')
print(f'Ячеек: {len(cells)}')
print('Продолжаю создание...')
