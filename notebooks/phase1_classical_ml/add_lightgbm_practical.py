#!/usr/bin/env python3
"""
Добавление практической части в LightGBM notebook
"""

import json

# Читаем текущий ноутбук
notebook_path = '02_lightgbm_deep_dive.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Новые практические ячейки
practical_cells = []

# ============================================================================
# HYPERPARAMETERS COMPARISON
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.7 Ключевые гиперпараметры LightGBM\n",
        "\n",
        "#### Сравнение с XGBoost\n",
        "\n",
        "| Концепция | XGBoost | LightGBM | Рекомендации |\n",
        "|-----------|---------|----------|-------------|\n",
        "| **Tree structure** | `max_depth` | `num_leaves` | LightGBM: 31 (default), XGBoost: 6 |\n",
        "| **Learning rate** | `learning_rate` | `learning_rate` | 0.01-0.1 |\n",
        "| **Regularization** | `lambda`, `alpha` | `lambda_l1`, `lambda_l2` | Аналогично |\n",
        "| **Sampling** | `subsample`, `colsample_bytree` | `bagging_fraction`, `feature_fraction` | 0.7-1.0 |\n",
        "| **Min data** | `min_child_weight` | `min_data_in_leaf` | LightGBM: 20, XGBoost: 1 |\n",
        "| **Binning** | `max_bin` | `max_bin` | 255 (default) |\n",
        "\n",
        "#### Специфичные для LightGBM\n",
        "\n",
        "| Параметр | Описание | Значения | Влияние |\n",
        "|----------|----------|----------|--------|\n",
        "| `num_leaves` | Количество листьев (не глубина!) | 31 (default), 15-255 | ↑ leaves → ↑ complexity |\n",
        "| `min_data_in_leaf` | Минимум примеров в листе | 20 (default), 10-100 | ↑ → более консервативные splits |\n",
        "| `max_depth` | Ограничение глубины (опционально) | -1 (no limit), 3-12 | Защита от overfitting leaf-wise |\n",
        "| `bagging_fraction` | Доля примеров для bagging | 1.0 (default), 0.5-1.0 | < 1 → ↓ overfitting |\n",
        "| `bagging_freq` | Частота bagging | 0 (disabled), 1-10 | Вместе с bagging_fraction |\n",
        "| `feature_fraction` | Доля признаков | 1.0 (default), 0.5-1.0 | Random Forest style |\n",
        "| `lambda_l1`, `lambda_l2` | L1/L2 регуляризация | 0 (default), 0-100 | Penalize большие веса |\n",
        "| `min_gain_to_split` | Минимальный gain | 0 (default), 0-1 | Аналог `gamma` в XGBoost |\n",
        "| `max_bin` | Количество bins для histogram | 255 (default), 63-511 | ↑ bins → ↑ точность, ↓ скорость |\n",
        "| `categorical_feature` | Список категориальных признаков | [] | Включает native categorical support |\n",
        "\n",
        "#### Параметры для ускорения\n",
        "\n",
        "| Параметр | Описание | Рекомендации |\n",
        "|----------|----------|-------------|\n",
        "| `num_threads` | Количество потоков | -1 (все CPU) |\n",
        "| `device_type` | CPU или GPU | 'cpu', 'gpu' |\n",
        "| `histogram_pool_size` | Размер cache для histogram | -1 (auto) |\n",
        "\n",
        "#### Стратегия тюнинга\n",
        "\n",
        "**Этап 1: Baseline (fast)**\n",
        "```python\n",
        "params = {\n",
        "    'objective': 'binary',\n",
        "    'metric': 'auc',\n",
        "    'num_leaves': 31,\n",
        "    'learning_rate': 0.1,\n",
        "    'n_estimators': 100\n",
        "}\n",
        "```\n",
        "\n",
        "**Этап 2: Tune structure**\n",
        "- Оптимизируем: `num_leaves`, `min_data_in_leaf`, `max_depth`\n",
        "- Цель: Найти правильную complexity\n",
        "\n",
        "**Этап 3: Sampling & Regularization**\n",
        "- Добавляем: `bagging_fraction`, `feature_fraction`\n",
        "- Tune: `lambda_l1`, `lambda_l2`, `min_gain_to_split`\n",
        "\n",
        "**Этап 4: Fine-tune learning**\n",
        "- Снижаем `learning_rate` до 0.01-0.05\n",
        "- Увеличиваем `n_estimators`\n",
        "- Используем early stopping\n",
        "\n",
        "---"
    ]
})

# ============================================================================
# COMPARISON TABLE
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.8 LightGBM vs XGBoost: Детальное сравнение\n",
        "\n",
        "| Аспект | XGBoost | LightGBM | Победитель |\n",
        "|--------|---------|----------|------------|\n",
        "| **Split finding** | Pre-sorted or histogram | Histogram-based | ⚡ LightGBM |\n",
        "| **Tree growth** | Level-wise | Leaf-wise (best-first) | 🎯 LightGBM (качество) |\n",
        "| **Sampling** | Random | GOSS (gradient-based) | 🧠 LightGBM |\n",
        "| **Feature bundling** | Нет | EFB (sparse features) | 📦 LightGBM |\n",
        "| **Categorical features** | One-hot нужен | Native support | 🏆 LightGBM |\n",
        "| **Скорость (CPU)** | Средняя | Очень быстрая | ⚡⚡ LightGBM |\n",
        "| **Скорость (GPU)** | Хорошая | Отличная | ⚡ LightGBM |\n",
        "| **Память** | Средняя | Низкая (histogram) | 💾 LightGBM |\n",
        "| **Точность (малые данные)** | Отлично | Хорошо | 🎯 XGBoost |\n",
        "| **Точность (большие данные)** | Хорошо | Отлично | 🎯 LightGBM |\n",
        "| **Overfitting** | Меньше склонен | Больше склонен (leaf-wise) | ✅ XGBoost |\n",
        "| **Стабильность** | Очень стабильна | Требует tuning | ✅ XGBoost |\n",
        "| **Настройка** | Проще (level-wise безопаснее) | Сложнее (leaf-wise требует care) | ✅ XGBoost |\n",
        "| **Зрелость** | Старше, больше ecosystem | Моложе, активное развитие | ✅ XGBoost |\n",
        "| **Документация** | Отличная | Хорошая | ✅ XGBoost |\n",
        "\n",
        "#### Когда использовать LightGBM?\n",
        "\n",
        "✅ **Используйте LightGBM если:**\n",
        "1. **Большие данные:** >10M примеров или >1000 признаков\n",
        "2. **Скорость критична:** Нужно быстрое обучение и inference\n",
        "3. **Категориальные признаки:** Много категорий с high cardinality\n",
        "4. **Разреженные данные:** One-hot encoded признаки, sparse матрицы\n",
        "5. **Ограниченная память:** Не хватает RAM для XGBoost\n",
        "6. **Готовы тюнить:** Есть время для подбора гиперпараметров\n",
        "\n",
        "✅ **Используйте XGBoost если:**\n",
        "1. **Малые/средние данные:** <1M примеров\n",
        "2. **Нужна стабильность:** Меньше риск overfitting\n",
        "3. **Первая модель:** Хотите baseline без complex tuning\n",
        "4. **Production-critical:** Зрелая, проверенная система\n",
        "5. **Плотные признаки:** Continuous features без sparsity\n",
        "\n",
        "#### Эмпирические результаты (Kaggle, research)\n",
        "\n",
        "**Скорость:**\n",
        "- LightGBM обычно **5-20x быстрее** на больших данных\n",
        "- На малых данных (<100k) разница незначительна\n",
        "\n",
        "**Качество:**\n",
        "- На больших данных: LightGBM часто **на 0.5-2% лучше** ROC-AUC\n",
        "- На малых данных: сопоставимо или XGBoost чуть лучше\n",
        "- **Лучшее решение:** Ensemble LightGBM + XGBoost!\n",
        "\n",
        "**Память:**\n",
        "- LightGBM: **~50% меньше** потребление памяти (histogram)\n",
        "\n",
        "---\n",
        "\n",
        "## Теоретическая часть завершена! Переходим к практике 🚀"
    ]
})

# ============================================================================
# PRACTICAL PART: IMPORTS
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📊 Часть 2: Практическая реализация\n",
        "\n",
        "### 2.1 Импорт библиотек"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Основные библиотеки\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from scipy import stats\n",
        "import warnings\n",
        "import time\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# LightGBM\n",
        "import lightgbm as lgb\n",
        "from lightgbm import LGBMClassifier\n",
        "\n",
        "# XGBoost для сравнения\n",
        "import xgboost as xgb\n",
        "from xgboost import XGBClassifier\n",
        "\n",
        "# Sklearn\n",
        "from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold\n",
        "from sklearn.model_selection import GridSearchCV, RandomizedSearchCV\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "from sklearn.metrics import (\n",
        "    accuracy_score, precision_score, recall_score, f1_score,\n",
        "    roc_auc_score, average_precision_score,\n",
        "    confusion_matrix, classification_report,\n",
        "    roc_curve, precision_recall_curve\n",
        ")\n",
        "\n",
        "# Baseline модели\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "# Настройка визуализации\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "sns.set_palette('husl')\n",
        "%matplotlib inline\n",
        "\n",
        "# Seed\n",
        "RANDOM_STATE = 42\n",
        "np.random.seed(RANDOM_STATE)\n",
        "\n",
        "print('✅ Библиотеки загружены')\n",
        "print(f'LightGBM version: {lgb.__version__}')\n",
        "print(f'XGBoost version: {xgb.__version__}')"
    ]
})

# ============================================================================
# DATA LOADING
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.2 Загрузка данных: Telco Customer Churn\n",
        "\n",
        "**Датасет:** IBM Telco Customer Churn\n",
        "\n",
        "**Источник:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n",
        "\n",
        "**Описание:**\n",
        "- ~7000 клиентов телеком компании\n",
        "- 20 признаков (демография, услуги, контракт, billing)\n",
        "- Целевая переменная: Churn (Yes/No)\n",
        "\n",
        "**Признаки:**\n",
        "\n",
        "**Demographic:**\n",
        "- `gender`: Male/Female\n",
        "- `SeniorCitizen`: 0/1\n",
        "- `Partner`: Yes/No (есть ли партнер)\n",
        "- `Dependents`: Yes/No (есть ли иждивенцы)\n",
        "\n",
        "**Services:**\n",
        "- `PhoneService`: Yes/No\n",
        "- `MultipleLines`: Yes/No/No phone service\n",
        "- `InternetService`: DSL/Fiber optic/No\n",
        "- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`: Yes/No/No internet\n",
        "\n",
        "**Account:**\n",
        "- `tenure`: Количество месяцев с компанией\n",
        "- `Contract`: Month-to-month / One year / Two year\n",
        "- `PaperlessBilling`: Yes/No\n",
        "- `PaymentMethod`: Electronic check / Mailed check / Bank transfer / Credit card\n",
        "- `MonthlyCharges`: Месячный платеж\n",
        "- `TotalCharges`: Общая сумма оплат\n",
        "\n",
        "**Target:**\n",
        "- `Churn`: Yes/No"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Загрузка данных\n",
        "import os\n",
        "\n",
        "data_path = '../../data/telco_churn.csv'\n",
        "\n",
        "if not os.path.exists(data_path):\n",
        "    print('❌ Файл не найден!')\n",
        "    print('Скачайте: https://www.kaggle.com/datasets/blastchar/telco-customer-churn')\n",
        "    print('Сохраните как: data/telco_churn.csv')\n",
        "else:\n",
        "    df = pd.read_csv(data_path)\n",
        "    print(f'✅ Данные загружены')\n",
        "    print(f'Размер: {df.shape[0]:,} строк, {df.shape[1]} столбцов')\n",
        "    print(f'Память: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')"
    ]
})

# Добавляем все практические ячейки
for cell in practical_cells:
    notebook['cells'].append(cell)

# Сохраняем
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Добавлено {len(practical_cells)} ячеек')
print(f'Всего ячеек в ноутбуке: {len(notebook["cells"])}')
