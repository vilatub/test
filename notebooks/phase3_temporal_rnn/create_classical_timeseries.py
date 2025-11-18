#!/usr/bin/env python3
"""
Создание полного ноутбука: Classical Time Series Analysis
Phase 3, Step 1: ARIMA, SARIMA, Prophet
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
        "# 📈 Classical Time Series Analysis\n",
        "\n",
        "**Phase 3: Temporal Data & RNN - Step 1**\n",
        "\n",
        "---\n",
        "\n",
        "## 🎯 Цели ноутбука\n",
        "\n",
        "1. **Понять основы временных рядов:** стационарность, тренд, сезонность\n",
        "2. **Классические модели:** ARIMA, SARIMA, Prophet\n",
        "3. **Сравнить подходы** и понять, когда какой метод работает лучше\n",
        "4. **Подготовка к Deep Learning:** RNN/LSTM будут в следующих шагах\n",
        "\n",
        "---\n",
        "\n",
        "## 📊 Датасет: Airline Passengers\n",
        "\n",
        "**Контекст:** Ежемесячное количество пассажиров авиалиний (1949-1960).\n",
        "\n",
        "**Почему этот датасет?**\n",
        "- 📊 **Классический пример** для обучения временным рядам\n",
        "- 📈 **Четкий тренд:** рост во времени\n",
        "- 🔄 **Сезонность:** годичные циклы\n",
        "- 🧪 **Нестационарность:** требует преобразований\n",
        "\n",
        "**Методы:**\n",
        "1. **ARIMA:** AutoRegressive Integrated Moving Average\n",
        "2. **SARIMA:** Seasonal ARIMA для учета сезонности\n",
        "3. **Prophet:** Автоматическая модель от Facebook\n",
        "\n",
        "---"
    ]
})

# ============================================================================
# THEORY PART 1: TIME SERIES BASICS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📚 Часть 1: Теория временных рядов\n",
        "\n",
        "### 1.1 Что такое временной ряд?\n",
        "\n",
        "**Временной ряд** — последовательность наблюдений $\\{y_1, y_2, ..., y_T\\}$, упорядоченных во времени.\n",
        "\n",
        "**Ключевое отличие от табличных данных:**\n",
        "- ✅ **Порядок важен** (нельзя перемешивать строки)\n",
        "- ✅ **Временные зависимости** (прошлое влияет на будущее)\n",
        "- ✅ **Автокорреляция** (значения коррелируют с собой в прошлом)\n",
        "\n",
        "---\n",
        "\n",
        "### 1.2 Компоненты временного ряда\n",
        "\n",
        "Временной ряд можно разложить на компоненты:\n",
        "\n",
        "$$y_t = T_t + S_t + R_t$$\n",
        "\n",
        "где:\n",
        "- $T_t$ — **Тренд** (долгосрочное направление)\n",
        "- $S_t$ — **Сезонность** (повторяющиеся паттерны)\n",
        "- $R_t$ — **Остатки** (случайный шум)\n",
        "\n",
        "**Альтернативно (мультипликативная модель):**\n",
        "\n",
        "$$y_t = T_t \\times S_t \\times R_t$$\n",
        "\n",
        "---\n",
        "\n",
        "### 1.3 Стационарность\n",
        "\n",
        "**Стационарный временной ряд** имеет:\n",
        "1. **Постоянное среднее:** $E[y_t] = \\mu$ для всех $t$\n",
        "2. **Постоянная дисперсия:** $Var(y_t) = \\sigma^2$ для всех $t$\n",
        "3. **Автокорреляция зависит только от лага:** $Cov(y_t, y_{t-k})$ зависит от $k$, но не от $t$\n",
        "\n",
        "**Почему важна?** Большинство моделей (ARIMA) требуют стационарности.\n",
        "\n",
        "**Как проверить?**\n",
        "- **Визуально:** график ряда\n",
        "- **ADF test (Augmented Dickey-Fuller):** $H_0$: ряд нестационарный\n",
        "- **KPSS test:** $H_0$: ряд стационарный\n",
        "\n",
        "**Как сделать стационарным?**\n",
        "- **Differencing:** $y'_t = y_t - y_{t-1}$ (устраняет тренд)\n",
        "- **Log transformation:** $y'_t = \\log(y_t)$ (стабилизирует дисперсию)\n",
        "- **Seasonal differencing:** $y'_t = y_t - y_{t-s}$ (устраняет сезонность)\n",
        "\n",
        "---"
    ]
})

# ============================================================================
# THEORY PART 2: ARIMA
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.4 ARIMA модель\n",
        "\n",
        "**ARIMA(p, d, q)** = **AR**(p) + **I**(d) + **MA**(q)\n",
        "\n",
        "#### AR (AutoRegressive) - авторегрессия\n",
        "\n",
        "Текущее значение зависит от прошлых значений:\n",
        "\n",
        "$$y_t = c + \\phi_1 y_{t-1} + \\phi_2 y_{t-2} + ... + \\phi_p y_{t-p} + \\varepsilon_t$$\n",
        "\n",
        "- $p$ — порядок AR (сколько прошлых значений используем)\n",
        "- $\\phi_i$ — коэффициенты\n",
        "\n",
        "**Пример AR(1):** $y_t = 0.8 y_{t-1} + \\varepsilon_t$ (80% от прошлого + шум)\n",
        "\n",
        "---\n",
        "\n",
        "#### I (Integrated) - интегрирование\n",
        "\n",
        "Сколько раз нужно применить differencing для стационарности:\n",
        "\n",
        "$$y'_t = y_t - y_{t-1}$$\n",
        "\n",
        "- $d=0$ — ряд уже стационарный\n",
        "- $d=1$ — берем первую разность\n",
        "- $d=2$ — вторая разность (редко)\n",
        "\n",
        "---\n",
        "\n",
        "#### MA (Moving Average) - скользящее среднее\n",
        "\n",
        "Текущее значение зависит от прошлых ошибок:\n",
        "\n",
        "$$y_t = \\mu + \\varepsilon_t + \\theta_1 \\varepsilon_{t-1} + \\theta_2 \\varepsilon_{t-2} + ... + \\theta_q \\varepsilon_{t-q}$$\n",
        "\n",
        "- $q$ — порядок MA\n",
        "- $\\theta_i$ — коэффициенты\n",
        "\n",
        "**Пример MA(1):** $y_t = \\varepsilon_t + 0.5 \\varepsilon_{t-1}$\n",
        "\n",
        "---\n",
        "\n",
        "#### Полная ARIMA(p, d, q)\n",
        "\n",
        "$$\n",
        "\\left(1 - \\sum_{i=1}^{p} \\phi_i L^i \\right) (1-L)^d y_t = \\left(1 + \\sum_{i=1}^{q} \\theta_i L^i \\right) \\varepsilon_t\n",
        "$$\n",
        "\n",
        "где $L$ — lag operator: $L y_t = y_{t-1}$\n",
        "\n",
        "**Как выбрать p, d, q?**\n",
        "1. **d:** ADF test (пока ряд не станет стационарным)\n",
        "2. **p:** ACF plot (автокорреляционная функция)\n",
        "3. **q:** PACF plot (частная автокорреляция)\n",
        "4. **Auto ARIMA:** автоматический перебор с AIC/BIC\n",
        "\n",
        "---"
    ]
})

# ============================================================================
# THEORY PART 3: SARIMA & PROPHET
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.5 SARIMA - Seasonal ARIMA\n",
        "\n",
        "**SARIMA(p, d, q)(P, D, Q, s)** добавляет сезонные компоненты:\n",
        "\n",
        "- **(p, d, q):** обычные ARIMA параметры\n",
        "- **(P, D, Q, s):** сезонные параметры\n",
        "  - $P$ — сезонная AR\n",
        "  - $D$ — сезонная разность\n",
        "  - $Q$ — сезонная MA\n",
        "  - $s$ — период сезонности (12 для месячных данных с годовой сезонностью)\n",
        "\n",
        "**Математически:**\n",
        "\n",
        "$$\\Phi_P(L^s) \\phi_p(L) \\nabla^D_s \\nabla^d y_t = \\Theta_Q(L^s) \\theta_q(L) \\varepsilon_t$$\n",
        "\n",
        "где:\n",
        "- $\\nabla^d$ — обычная разность\n",
        "- $\\nabla^D_s$ — сезонная разность: $y_t - y_{t-s}$\n",
        "\n",
        "**Пример:** SARIMA(1, 1, 1)(1, 1, 1, 12)\n",
        "- AR(1), MA(1), differencing=1\n",
        "- Seasonal AR(1), MA(1), seasonal differencing=1, период=12\n",
        "\n",
        "---\n",
        "\n",
        "### 1.6 Prophet (Facebook)\n",
        "\n",
        "**Prophet** — аддитивная модель:\n",
        "\n",
        "$$y_t = g(t) + s(t) + h(t) + \\varepsilon_t$$\n",
        "\n",
        "где:\n",
        "- $g(t)$ — **тренд** (кусочно-линейный или логистический рост)\n",
        "- $s(t)$ — **сезонность** (Fourier series для годовой/недельной)\n",
        "- $h(t)$ — **праздники** и события\n",
        "- $\\varepsilon_t$ — шум\n",
        "\n",
        "**Преимущества Prophet:**\n",
        "- ✅ Автоматический (мало параметров)\n",
        "- ✅ Справляется с пропусками\n",
        "- ✅ Учитывает праздники\n",
        "- ✅ Интерпретируемые компоненты\n",
        "- ✅ Робастен к выбросам\n",
        "\n",
        "**Когда использовать:**\n",
        "- 📊 Бизнес-метрики (продажи, трафик)\n",
        "- 🔄 Четкая сезонность (годовая/недельная)\n",
        "- 🎉 Влияние праздников\n",
        "- ⏱️ Ежедневные/недельные данные\n",
        "\n",
        "---"
    ]
})

# ============================================================================
# PRACTICAL PART: IMPORTS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 💻 Часть 2: Практика\n",
        "\n",
        "### 2.1 Импорт библиотек"
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
        "# Для временных рядов\n",
        "from statsmodels.tsa.seasonal import seasonal_decompose\n",
        "from statsmodels.tsa.stattools import adfuller, kpss\n",
        "from statsmodels.tsa.arima.model import ARIMA\n",
        "from statsmodels.tsa.statespace.sarimax import SARIMAX\n",
        "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\n",
        "\n",
        "# Prophet\n",
        "try:\n",
        "    from prophet import Prophet\n",
        "except ImportError:\n",
        "    print(\"Prophet not installed. Install: pip install prophet\")\n",
        "\n",
        "# Метрики\n",
        "from sklearn.metrics import mean_squared_error, mean_absolute_error\n",
        "\n",
        "# Настройка визуализации\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "sns.set_palette(\"husl\")\n",
        "%matplotlib inline\n",
        "\n",
        "print(\"✅ Библиотеки загружены\")"
    ]
})

# ============================================================================
# LOAD DATA
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.2 Загрузка датасета Airline Passengers\n",
        "\n",
        "Датасет доступен напрямую через seaborn или statsmodels."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Загрузка датасета\n",
        "try:\n",
        "    # Через statsmodels\n",
        "    from statsmodels.datasets import get_rdataset\n",
        "    airline_data = get_rdataset('AirPassengers', 'datasets')\n",
        "    df = airline_data.data\n",
        "    df.columns = ['time', 'passengers']\n",
        "except:\n",
        "    # Альтернативный способ\n",
        "    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'\n",
        "    df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')\n",
        "    df.columns = ['passengers']\n",
        "    df.reset_index(inplace=True)\n",
        "    df.columns = ['time', 'passengers']\n",
        "\n",
        "# Преобразуем time в datetime\n",
        "df['time'] = pd.to_datetime(df['time'])\n",
        "df.set_index('time', inplace=True)\n",
        "\n",
        "print(f\"Датасет загружен: {df.shape[0]} наблюдений\")\n",
        "print(f\"Период: {df.index.min()} - {df.index.max()}\")\n",
        "print(f\"\\nПервые строки:\")\n",
        "df.head()"
    ]
})

# ============================================================================
# EDA
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.3 Exploratory Data Analysis (EDA)"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Базовая статистика\n",
        "print(\"Статистика временного ряда:\")\n",
        "print(df['passengers'].describe())\n",
        "\n",
        "# График временного ряда\n",
        "plt.figure(figsize=(14, 5))\n",
        "plt.plot(df.index, df['passengers'], linewidth=2)\n",
        "plt.title('Airline Passengers (1949-1960)', fontsize=16, fontweight='bold')\n",
        "plt.xlabel('Time', fontsize=12)\n",
        "plt.ylabel('Number of Passengers', fontsize=12)\n",
        "plt.grid(alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n📊 Наблюдения:\")\n",
        "print(\"1. ✅ Четкий восходящий тренд\")\n",
        "print(\"2. ✅ Годовая сезонность (пики летом)\")\n",
        "print(\"3. ✅ Увеличивающаяся амплитуда сезонности (мультипликативная модель)\")\n",
        "print(\"4. ❌ Ряд НЕстационарный (требует преобразований)\")"
    ]
})

# ============================================================================
# DECOMPOSITION
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.4 Декомпозиция временного ряда\n",
        "\n",
        "Разложим ряд на **тренд**, **сезонность** и **остатки**."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Декомпозиция (мультипликативная модель)\n",
        "decomposition = seasonal_decompose(df['passengers'], model='multiplicative', period=12)\n",
        "\n",
        "# Визуализация\n",
        "fig, axes = plt.subplots(4, 1, figsize=(14, 10))\n",
        "\n",
        "# Исходный ряд\n",
        "decomposition.observed.plot(ax=axes[0], color='blue')\n",
        "axes[0].set_title('Original Time Series', fontsize=14, fontweight='bold')\n",
        "axes[0].set_ylabel('Passengers')\n",
        "\n",
        "# Тренд\n",
        "decomposition.trend.plot(ax=axes[1], color='green')\n",
        "axes[1].set_title('Trend Component', fontsize=14, fontweight='bold')\n",
        "axes[1].set_ylabel('Trend')\n",
        "\n",
        "# Сезонность\n",
        "decomposition.seasonal.plot(ax=axes[2], color='orange')\n",
        "axes[2].set_title('Seasonal Component (Period=12 months)', fontsize=14, fontweight='bold')\n",
        "axes[2].set_ylabel('Seasonality')\n",
        "\n",
        "# Остатки\n",
        "decomposition.resid.plot(ax=axes[3], color='red')\n",
        "axes[3].set_title('Residuals (Random Noise)', fontsize=14, fontweight='bold')\n",
        "axes[3].set_ylabel('Residuals')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"✅ Декомпозиция показывает:\")\n",
        "print(\"  - Линейный возрастающий тренд\")\n",
        "print(\"  - Стабильную годовую сезонность (пики летом)\")\n",
        "print(\"  - Остатки близки к белому шуму\")"
    ]
})

# ============================================================================
# STATIONARITY TESTS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.5 Тесты на стационарность\n",
        "\n",
        "**ADF test:** $H_0$ = ряд нестационарный (есть unit root)  \n",
        "**KPSS test:** $H_0$ = ряд стационарный\n",
        "\n",
        "Для стационарности нужно:\n",
        "- ADF: p-value < 0.05 (отвергаем $H_0$)\n",
        "- KPSS: p-value > 0.05 (не отвергаем $H_0$)"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def check_stationarity(timeseries, title):\n",
        "    \"\"\"Проверка стационарности с ADF и KPSS тестами\"\"\"\n",
        "    print(f\"\\n{'='*60}\")\n",
        "    print(f\"Тест стационарности: {title}\")\n",
        "    print(f\"{'='*60}\")\n",
        "    \n",
        "    # ADF Test\n",
        "    adf_result = adfuller(timeseries.dropna(), autolag='AIC')\n",
        "    print(f\"\\nADF Test (H0: нестационарный):\")\n",
        "    print(f\"  ADF Statistic: {adf_result[0]:.4f}\")\n",
        "    print(f\"  p-value: {adf_result[1]:.4f}\")\n",
        "    print(f\"  Critical values: {adf_result[4]}\")\n",
        "    \n",
        "    if adf_result[1] < 0.05:\n",
        "        print(f\"  ✅ Ряд СТАЦИОНАРНЫЙ (p < 0.05, отвергаем H0)\")\n",
        "    else:\n",
        "        print(f\"  ❌ Ряд НЕСТАЦИОНАРНЫЙ (p >= 0.05, не отвергаем H0)\")\n",
        "    \n",
        "    # KPSS Test\n",
        "    kpss_result = kpss(timeseries.dropna(), regression='c', nlags='auto')\n",
        "    print(f\"\\nKPSS Test (H0: стационарный):\")\n",
        "    print(f\"  KPSS Statistic: {kpss_result[0]:.4f}\")\n",
        "    print(f\"  p-value: {kpss_result[1]:.4f}\")\n",
        "    print(f\"  Critical values: {kpss_result[3]}\")\n",
        "    \n",
        "    if kpss_result[1] > 0.05:\n",
        "        print(f\"  ✅ Ряд СТАЦИОНАРНЫЙ (p > 0.05, не отвергаем H0)\")\n",
        "    else:\n",
        "        print(f\"  ❌ Ряд НЕСТАЦИОНАРНЫЙ (p <= 0.05, отвергаем H0)\")\n",
        "\n",
        "# Проверка оригинального ряда\n",
        "check_stationarity(df['passengers'], \"Original Series\")"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.6 Преобразование к стационарности\n",
        "\n",
        "Применим **log transformation** + **differencing**:"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Log transformation для стабилизации дисперсии\n",
        "df['log_passengers'] = np.log(df['passengers'])\n",
        "\n",
        "# First differencing для устранения тренда\n",
        "df['log_diff'] = df['log_passengers'].diff()\n",
        "\n",
        "# Сезонная разность\n",
        "df['log_seasonal_diff'] = df['log_passengers'].diff(12)\n",
        "\n",
        "# Визуализация преобразований\n",
        "fig, axes = plt.subplots(3, 1, figsize=(14, 10))\n",
        "\n",
        "# Log transform\n",
        "df['log_passengers'].plot(ax=axes[0], color='blue')\n",
        "axes[0].set_title('Log Transformation', fontsize=14, fontweight='bold')\n",
        "axes[0].set_ylabel('log(Passengers)')\n",
        "\n",
        "# First difference\n",
        "df['log_diff'].plot(ax=axes[1], color='green')\n",
        "axes[1].set_title('Log + First Difference', fontsize=14, fontweight='bold')\n",
        "axes[1].set_ylabel('Δlog(Passengers)')\n",
        "axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)\n",
        "\n",
        "# Seasonal difference\n",
        "df['log_seasonal_diff'].plot(ax=axes[2], color='orange')\n",
        "axes[2].set_title('Log + Seasonal Difference (lag=12)', fontsize=14, fontweight='bold')\n",
        "axes[2].set_ylabel('Δ12 log(Passengers)')\n",
        "axes[2].axhline(0, color='red', linestyle='--', alpha=0.5)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Проверка стационарности после преобразований\n",
        "check_stationarity(df['log_diff'], \"Log + First Difference\")"
    ]
})

# ============================================================================
# ACF / PACF
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.7 ACF и PACF графики\n",
        "\n",
        "**ACF (AutoCorrelation Function):** корреляция с прошлыми значениями  \n",
        "**PACF (Partial AutoCorrelation Function):** корреляция с учетом промежуточных значений\n",
        "\n",
        "**Правила:**\n",
        "- **ACF постепенно затухает, PACF резко обрывается** → AR(p), где p = обрыв на PACF\n",
        "- **PACF постепенно затухает, ACF резко обрывается** → MA(q), где q = обрыв на ACF\n",
        "- **Оба затухают** → ARMA(p, q)"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ACF и PACF графики\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "plot_acf(df['log_diff'].dropna(), lags=40, ax=axes[0])\n",
        "axes[0].set_title('ACF (AutoCorrelation Function)', fontsize=14, fontweight='bold')\n",
        "\n",
        "plot_pacf(df['log_diff'].dropna(), lags=40, ax=axes[1])\n",
        "axes[1].set_title('PACF (Partial AutoCorrelation)', fontsize=14, fontweight='bold')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n📊 Интерпретация:\")\n",
        "print(\"  - ACF: значимые лаги на 1, 12, 24 (сезонность)\")\n",
        "print(\"  - PACF: значимые лаги на 1, 12 (AR компоненты)\")\n",
        "print(\"  - Предположение: ARIMA(1, 1, 1) + сезонная компонента (12)\")"
    ]
})

# Добавляем остальные ячейки (продолжение следует...)
notebook['cells'] = cells

# Сохраняем (пока частично, продолжим в следующей части)
output_path = '/home/user/test/notebooks/phase3_temporal_rnn/01_classical_timeseries.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Часть 1 создана: {output_path}')
print(f'Ячеек создано: {len(cells)}')
print('Следующая часть: ARIMA, SARIMA, Prophet модели...')
