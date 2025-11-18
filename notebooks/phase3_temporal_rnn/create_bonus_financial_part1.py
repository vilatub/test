#!/usr/bin/env python3
"""
Phase 3 BONUS: Real-world Financial Pattern Recognition
Part 1: Introduction, Data Loading, Feature Engineering
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
        "# 💰 Real-world Financial Pattern Recognition\n",
        "\n",
        "**Phase 3 BONUS: Advanced Multivariate Time Series**\n",
        "\n",
        "---\n",
        "\n",
        "## 🎯 Зачем этот ноутбук?\n",
        "\n",
        "В основных ноутбуках Phase 3 мы использовали **Airline Passengers:**\n",
        "- ❌ Маленький датасет (144 точки)\n",
        "- ❌ Univariate (одна переменная)\n",
        "- ❌ Простые паттерны (линейный тренд + сезонность)\n",
        "- ❌ Не показывает силу Deep Learning\n",
        "\n",
        "**Этот ноутбук:**\n",
        "- ✅ **Real-world датасет:** EURUSD 4H, 5 лет (>10,000 точек)\n",
        "- ✅ **Multivariate:** 20+ технических индикаторов\n",
        "- ✅ **Сложные паттерны:** графические формации (голова-плечи, флаги, треугольники)\n",
        "- ✅ **Три задачи:** цена, направление, паттерн-распознавание\n",
        "- ✅ **Практика:** как формализовать \"неформализуемые\" паттерны\n",
        "\n",
        "---\n",
        "\n",
        "## 📊 Три задачи разной сложности\n",
        "\n",
        "### Task 1: Price Forecasting (Регрессия)\n",
        "**Задача:** Предсказать цену закрытия через N свечей  \n",
        "**Базовая метрика:** RMSE, MAE  \n",
        "**Зачем:** Baseline для сравнения\n",
        "\n",
        "### Task 2: Direction Classification (Классификация)\n",
        "**Задача:** Предсказать направление (UP/DOWN/NEUTRAL)  \n",
        "**Метрики:** Accuracy, Precision, Recall, F1  \n",
        "**Зачем:** Более практично для трейдинга\n",
        "\n",
        "### Task 3: Pattern Recognition ⭐ (ГЛАВНЫЙ ФОКУС)\n",
        "**Задача:** Распознать графические и индикаторные паттерны  \n",
        "**Паттерны:**\n",
        "- **Indicator-based:** RSI divergence, MACD crossover, Bollinger Squeeze\n",
        "- **Chart patterns:** Head & Shoulders, Double Top/Bottom, Flags, Triangles\n",
        "- **Breakouts:** Support/Resistance пробои с подтверждением\n",
        "\n",
        "**Метод:**  \n",
        "1. **Feature engineering** подталкивает сеть думать в нужном направлении\n",
        "2. **LSTM + Attention** учится комбинировать фичи\n",
        "3. **SHAP** показывает, какие фичи сработали\n",
        "4. **Итерация:** добавляем/убираем фичи на основе результатов\n",
        "\n",
        "---\n",
        "\n",
        "## ⚠️ DISCLAIMER\n",
        "\n",
        "**Этот ноутбук создан в ОБРАЗОВАТЕЛЬНЫХ целях.**\n",
        "\n",
        "- 📚 Демонстрирует применение Deep Learning на реальных данных\n",
        "- 🔬 Показывает workflow feature engineering для паттернов\n",
        "- ❌ **НЕ является торговой рекомендацией**\n",
        "- ❌ **НЕ гарантирует прибыль**\n",
        "- ⚠️ Прошлые результаты не гарантируют будущее\n",
        "- ⚠️ Финансовые рынки эффективны (теория EMH)\n",
        "- ⚠️ Реальный трейдинг требует risk management, комиссии, проскальзывание\n",
        "\n",
        "**Используйте полученные знания ответственно!**\n",
        "\n",
        "---"
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
        "from datetime import datetime, timedelta\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Для загрузки данных\n",
        "try:\n",
        "    import yfinance as yf\n",
        "    print(\"✅ yfinance доступен\")\n",
        "except ImportError:\n",
        "    print(\"⚠️ yfinance не установлен. Установка: pip install yfinance\")\n",
        "\n",
        "# Технические индикаторы\n",
        "try:\n",
        "    import ta\n",
        "    print(\"✅ ta (Technical Analysis library) доступен\")\n",
        "except ImportError:\n",
        "    print(\"⚠️ ta не установлен. Установка: pip install ta\")\n",
        "\n",
        "# PyTorch\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import torch.nn.functional as F\n",
        "from torch.utils.data import Dataset, DataLoader, TensorDataset\n",
        "\n",
        "# Sklearn\n",
        "from sklearn.preprocessing import MinMaxScaler, StandardScaler\n",
        "from sklearn.metrics import (\n",
        "    mean_squared_error, mean_absolute_error,\n",
        "    accuracy_score, precision_score, recall_score, f1_score,\n",
        "    classification_report, confusion_matrix\n",
        ")\n",
        "\n",
        "# Для интерпретации\n",
        "try:\n",
        "    import shap\n",
        "    print(\"✅ SHAP доступен\")\n",
        "except ImportError:\n",
        "    print(\"⚠️ SHAP не установлен. Установка: pip install shap\")\n",
        "\n",
        "# Настройки\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "sns.set_palette(\"husl\")\n",
        "%matplotlib inline\n",
        "\n",
        "# Device\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f\"\\nDevice: {device}\")\n",
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
        "### 1.2 Загрузка EURUSD данных\n",
        "\n",
        "**Источник:** Yahoo Finance (EURUSD=X)  \n",
        "**Timeframe:** 4H (через 1H с ресемплингом)  \n",
        "**Период:** 5 лет  \n",
        "\n",
        "**Почему EURUSD:**\n",
        "- ✅ Самая ликвидная валютная пара\n",
        "- ✅ Торгуется 24/5 (много данных)\n",
        "- ✅ Менее волатильна, чем акции (стабильнее паттерны)\n",
        "- ✅ Бесплатный доступ через yfinance"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Параметры загрузки\n",
        "TICKER = 'EURUSD=X'\n",
        "PERIOD = '5y'  # 5 лет данных\n",
        "INTERVAL = '1h'  # часовые данные (потом ресемплируем в 4H)\n",
        "\n",
        "print(f\"Загрузка {TICKER} за {PERIOD}...\")\n",
        "\n",
        "# Загрузка\n",
        "try:\n",
        "    df_raw = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)\n",
        "    print(f\"✅ Загружено {len(df_raw)} часовых свечей\")\n",
        "except Exception as e:\n",
        "    print(f\"❌ Ошибка загрузки: {e}\")\n",
        "    print(\"\\nАльтернатива: используем синтетические данные для демонстрации...\")\n",
        "    # Создаем синтетические данные для демонстрации\n",
        "    dates = pd.date_range(end=datetime.now(), periods=10000, freq='1H')\n",
        "    np.random.seed(42)\n",
        "    price = 1.08 + np.cumsum(np.random.randn(10000) * 0.0001)\n",
        "    df_raw = pd.DataFrame({\n",
        "        'Open': price + np.random.randn(10000) * 0.0001,\n",
        "        'High': price + abs(np.random.randn(10000) * 0.0002),\n",
        "        'Low': price - abs(np.random.randn(10000) * 0.0002),\n",
        "        'Close': price,\n",
        "        'Volume': np.random.randint(1000, 10000, 10000)\n",
        "    }, index=dates)\n",
        "\n",
        "# Ресемплинг в 4H\n",
        "df = df_raw.resample('4H').agg({\n",
        "    'Open': 'first',\n",
        "    'High': 'max',\n",
        "    'Low': 'min',\n",
        "    'Close': 'last',\n",
        "    'Volume': 'sum'\n",
        "}).dropna()\n",
        "\n",
        "print(f\"\\n4H ресемплинг: {len(df)} свечей\")\n",
        "print(f\"Период: {df.index.min()} - {df.index.max()}\")\n",
        "print(f\"\\nПервые строки:\")\n",
        "df.head()"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Визуализация сырых данных\n",
        "fig, axes = plt.subplots(2, 1, figsize=(16, 8))\n",
        "\n",
        "# Price\n",
        "axes[0].plot(df.index, df['Close'], linewidth=1.5, label='Close Price')\n",
        "axes[0].set_title('EURUSD 4H Chart (5 years)', fontsize=16, fontweight='bold')\n",
        "axes[0].set_ylabel('Price', fontsize=12)\n",
        "axes[0].legend()\n",
        "axes[0].grid(alpha=0.3)\n",
        "\n",
        "# Volume\n",
        "axes[1].bar(df.index, df['Volume'], width=0.1, alpha=0.6, label='Volume')\n",
        "axes[1].set_title('Volume', fontsize=14, fontweight='bold')\n",
        "axes[1].set_xlabel('Date', fontsize=12)\n",
        "axes[1].set_ylabel('Volume', fontsize=12)\n",
        "axes[1].legend()\n",
        "axes[1].grid(alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(f\"\\n📊 Датасет готов: {len(df)} 4H-свечей за 5 лет\")\n",
        "print(\"Это ~10,000 точек - достаточно для Deep Learning!\")"
    ]
})

# ============================================================================
# BASIC STATISTICS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.3 Базовая статистика и проверка данных"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Статистика\n",
        "print(\"Базовая статистика:\")\n",
        "print(df.describe())\n",
        "\n",
        "# Проверка пропусков\n",
        "print(f\"\\nПропуски:\")\n",
        "print(df.isnull().sum())\n",
        "\n",
        "# Удаляем пропуски если есть\n",
        "df = df.dropna()\n",
        "\n",
        "# Базовые метрики\n",
        "price_range = df['High'].max() - df['Low'].min()\n",
        "avg_candle_size = (df['High'] - df['Low']).mean()\n",
        "volatility = df['Close'].pct_change().std()\n",
        "\n",
        "print(f\"\\n📈 Анализ волатильности:\")\n",
        "print(f\"  Price range: {price_range:.5f}\")\n",
        "print(f\"  Avg candle size: {avg_candle_size:.5f}\")\n",
        "print(f\"  Volatility (std of returns): {volatility:.5f}\")\n",
        "\n",
        "# Returns distribution\n",
        "returns = df['Close'].pct_change().dropna()\n",
        "\n",
        "plt.figure(figsize=(12, 5))\n",
        "plt.hist(returns, bins=100, alpha=0.7, edgecolor='black')\n",
        "plt.title('Returns Distribution (4H)', fontsize=16, fontweight='bold')\n",
        "plt.xlabel('Returns', fontsize=12)\n",
        "plt.ylabel('Frequency', fontsize=12)\n",
        "plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')\n",
        "plt.legend()\n",
        "plt.grid(alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(f\"\\nReturns:\")\n",
        "print(f\"  Mean: {returns.mean():.6f}\")\n",
        "print(f\"  Std: {returns.std():.6f}\")\n",
        "print(f\"  Skew: {returns.skew():.3f}\")\n",
        "print(f\"  Kurtosis: {returns.kurtosis():.3f}\")"
    ]
})

# Сохраняем первую часть
notebook['cells'] = cells

output_path = '/home/user/test/notebooks/phase3_temporal_rnn/bonus_financial_patterns.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Part 1 создана: {output_path}')
print(f'Ячеек: {len(cells)}')
print('Следующая часть: Feature Engineering (технические индикаторы)...')
