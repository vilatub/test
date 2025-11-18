#!/usr/bin/env python3
"""
Добавление моделей ARIMA, SARIMA, Prophet в ноутбук Classical Time Series
"""

import json

# Загружаем существующий ноутбук
notebook_path = '/home/user/test/notebooks/phase3_temporal_rnn/01_classical_timeseries.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

cells = notebook['cells']

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.8 Train/Test Split\n",
        "\n",
        "**Важно:** Для временных рядов НЕ используем random shuffle!  \n",
        "Разделяем последовательно: train = первые 80%, test = последние 20%."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Разделение на train/test\n",
        "train_size = int(len(df) * 0.8)\n",
        "\n",
        "train = df['passengers'][:train_size]\n",
        "test = df['passengers'][train_size:]\n",
        "\n",
        "print(f\"Train size: {len(train)} ({len(train)/len(df)*100:.1f}%)\")\n",
        "print(f\"Test size: {len(test)} ({len(test)/len(df)*100:.1f}%)\")\n",
        "print(f\"\\nTrain: {train.index.min()} - {train.index.max()}\")\n",
        "print(f\"Test: {test.index.min()} - {test.index.max()}\")\n",
        "\n",
        "# Визуализация split\n",
        "plt.figure(figsize=(14, 5))\n",
        "plt.plot(train.index, train, label='Train', linewidth=2)\n",
        "plt.plot(test.index, test, label='Test', linewidth=2, color='orange')\n",
        "plt.axvline(train.index[-1], color='red', linestyle='--', alpha=0.5, label='Split')\n",
        "plt.title('Train/Test Split', fontsize=16, fontweight='bold')\n",
        "plt.xlabel('Time')\n",
        "plt.ylabel('Passengers')\n",
        "plt.legend()\n",
        "plt.grid(alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# ============================================================================
# ARIMA MODEL
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🔮 Часть 3: Модели прогнозирования\n",
        "\n",
        "### 3.1 ARIMA Model\n",
        "\n",
        "Используем **ARIMA(p, d, q)** с параметрами из ACF/PACF анализа.  \n",
        "Попробуем несколько вариантов и выберем лучший по AIC."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ARIMA(1, 1, 1) - базовая модель\n",
        "print(\"Обучение ARIMA(1, 1, 1)...\")\n",
        "arima_model = ARIMA(train, order=(1, 1, 1))\n",
        "arima_fitted = arima_model.fit()\n",
        "\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"ARIMA Model Summary\")\n",
        "print(\"=\"*60)\n",
        "print(arima_fitted.summary())\n",
        "\n",
        "# Прогноз на test\n",
        "arima_forecast = arima_fitted.forecast(steps=len(test))\n",
        "\n",
        "# Метрики\n",
        "arima_mse = mean_squared_error(test, arima_forecast)\n",
        "arima_rmse = np.sqrt(arima_mse)\n",
        "arima_mae = mean_absolute_error(test, arima_forecast)\n",
        "\n",
        "print(f\"\\n📊 ARIMA Metrics:\")\n",
        "print(f\"  RMSE: {arima_rmse:.2f}\")\n",
        "print(f\"  MAE: {arima_mae:.2f}\")\n",
        "print(f\"  AIC: {arima_fitted.aic:.2f}\")\n",
        "print(f\"  BIC: {arima_fitted.bic:.2f}\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Визуализация ARIMA predictions\n",
        "plt.figure(figsize=(14, 6))\n",
        "\n",
        "plt.plot(train.index, train, label='Train', linewidth=2)\n",
        "plt.plot(test.index, test, label='Test (Actual)', linewidth=2, color='orange')\n",
        "plt.plot(test.index, arima_forecast, label='ARIMA Forecast', linewidth=2, \n",
        "         color='green', linestyle='--')\n",
        "\n",
        "plt.axvline(train.index[-1], color='red', linestyle='--', alpha=0.5)\n",
        "plt.title('ARIMA(1, 1, 1) Forecast', fontsize=16, fontweight='bold')\n",
        "plt.xlabel('Time')\n",
        "plt.ylabel('Passengers')\n",
        "plt.legend()\n",
        "plt.grid(alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n⚠️ Проблема: ARIMA не улавливает сезонность!\")\n",
        "print(\"Прогноз линейный, без годовых циклов → нужен SARIMA\")"
    ]
})

# ============================================================================
# AUTO ARIMA
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.2 Auto ARIMA - автоматический подбор параметров\n",
        "\n",
        "Используем `pmdarima` для автоматического поиска лучших параметров:"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "try:\n",
        "    from pmdarima import auto_arima\n",
        "    \n",
        "    print(\"Запуск Auto ARIMA (может занять время)...\")\n",
        "    auto_model = auto_arima(\n",
        "        train, \n",
        "        start_p=0, start_q=0,\n",
        "        max_p=5, max_q=5,\n",
        "        d=None,  # автоматический выбор d\n",
        "        seasonal=False,  # пока без сезонности\n",
        "        stepwise=True,\n",
        "        suppress_warnings=True,\n",
        "        error_action='ignore',\n",
        "        trace=True\n",
        "    )\n",
        "    \n",
        "    print(\"\\n\" + \"=\"*60)\n",
        "    print(\"Best Auto ARIMA Model:\")\n",
        "    print(\"=\"*60)\n",
        "    print(auto_model.summary())\n",
        "    \n",
        "    # Прогноз\n",
        "    auto_forecast = auto_model.predict(n_periods=len(test))\n",
        "    \n",
        "    # Метрики\n",
        "    auto_rmse = np.sqrt(mean_squared_error(test, auto_forecast))\n",
        "    auto_mae = mean_absolute_error(test, auto_forecast)\n",
        "    \n",
        "    print(f\"\\n📊 Auto ARIMA Metrics:\")\n",
        "    print(f\"  RMSE: {auto_rmse:.2f}\")\n",
        "    print(f\"  MAE: {auto_mae:.2f}\")\n",
        "    \n",
        "except ImportError:\n",
        "    print(\"⚠️ pmdarima не установлен\")\n",
        "    print(\"Установка: pip install pmdarima\")\n",
        "    auto_forecast = None"
    ]
})

# ============================================================================
# SARIMA MODEL
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.3 SARIMA - Seasonal ARIMA\n",
        "\n",
        "Добавляем сезонную компоненту: **SARIMA(p, d, q)(P, D, Q, s)**  \n",
        "Для месячных данных с годовой сезонностью: s=12"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# SARIMA(1, 1, 1)(1, 1, 1, 12)\n",
        "print(\"Обучение SARIMA(1, 1, 1)(1, 1, 1, 12)...\")\n",
        "sarima_model = SARIMAX(\n",
        "    train,\n",
        "    order=(1, 1, 1),  # ARIMA часть\n",
        "    seasonal_order=(1, 1, 1, 12),  # Сезонная часть\n",
        "    enforce_stationarity=False,\n",
        "    enforce_invertibility=False\n",
        ")\n",
        "\n",
        "sarima_fitted = sarima_model.fit(disp=False)\n",
        "\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"SARIMA Model Summary\")\n",
        "print(\"=\"*60)\n",
        "print(sarima_fitted.summary())\n",
        "\n",
        "# Прогноз\n",
        "sarima_forecast = sarima_fitted.forecast(steps=len(test))\n",
        "\n",
        "# Метрики\n",
        "sarima_mse = mean_squared_error(test, sarima_forecast)\n",
        "sarima_rmse = np.sqrt(sarima_mse)\n",
        "sarima_mae = mean_absolute_error(test, sarima_forecast)\n",
        "\n",
        "print(f\"\\n📊 SARIMA Metrics:\")\n",
        "print(f\"  RMSE: {sarima_rmse:.2f}\")\n",
        "print(f\"  MAE: {sarima_mae:.2f}\")\n",
        "print(f\"  AIC: {sarima_fitted.aic:.2f}\")\n",
        "print(f\"  BIC: {sarima_fitted.bic:.2f}\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Визуализация SARIMA predictions\n",
        "plt.figure(figsize=(14, 6))\n",
        "\n",
        "plt.plot(train.index, train, label='Train', linewidth=2)\n",
        "plt.plot(test.index, test, label='Test (Actual)', linewidth=2, color='orange')\n",
        "plt.plot(test.index, sarima_forecast, label='SARIMA Forecast', linewidth=2, \n",
        "         color='purple', linestyle='--')\n",
        "\n",
        "plt.axvline(train.index[-1], color='red', linestyle='--', alpha=0.5)\n",
        "plt.title('SARIMA(1,1,1)(1,1,1,12) Forecast', fontsize=16, fontweight='bold')\n",
        "plt.xlabel('Time')\n",
        "plt.ylabel('Passengers')\n",
        "plt.legend()\n",
        "plt.grid(alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n✅ SARIMA захватывает сезонность!\")\n",
        "print(\"Прогноз учитывает годовые циклы (пики летом, спады зимой)\")"
    ]
})

# ============================================================================
# PROPHET MODEL
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.4 Prophet (Facebook)\n",
        "\n",
        "Prophet требует специальный формат данных: columns=['ds', 'y']"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "try:\n",
        "    # Подготовка данных для Prophet\n",
        "    train_prophet = pd.DataFrame({\n",
        "        'ds': train.index,\n",
        "        'y': train.values\n",
        "    })\n",
        "    \n",
        "    # Обучение Prophet\n",
        "    print(\"Обучение Prophet...\")\n",
        "    prophet_model = Prophet(\n",
        "        yearly_seasonality=True,\n",
        "        weekly_seasonality=False,\n",
        "        daily_seasonality=False,\n",
        "        seasonality_mode='multiplicative'  # из-за растущей амплитуды\n",
        "    )\n",
        "    \n",
        "    prophet_model.fit(train_prophet)\n",
        "    \n",
        "    # Прогноз\n",
        "    future = prophet_model.make_future_dataframe(periods=len(test), freq='MS')\n",
        "    prophet_forecast_full = prophet_model.predict(future)\n",
        "    \n",
        "    # Извлекаем прогноз для test периода\n",
        "    prophet_forecast = prophet_forecast_full['yhat'].iloc[-len(test):].values\n",
        "    \n",
        "    # Метрики\n",
        "    prophet_mse = mean_squared_error(test, prophet_forecast)\n",
        "    prophet_rmse = np.sqrt(prophet_mse)\n",
        "    prophet_mae = mean_absolute_error(test, prophet_forecast)\n",
        "    \n",
        "    print(f\"\\n📊 Prophet Metrics:\")\n",
        "    print(f\"  RMSE: {prophet_rmse:.2f}\")\n",
        "    print(f\"  MAE: {prophet_mae:.2f}\")\n",
        "    \n",
        "    prophet_available = True\n",
        "    \n",
        "except Exception as e:\n",
        "    print(f\"⚠️ Prophet error: {e}\")\n",
        "    print(\"Установка: pip install prophet\")\n",
        "    prophet_forecast = None\n",
        "    prophet_available = False"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "if prophet_available:\n",
        "    # Визуализация Prophet\n",
        "    plt.figure(figsize=(14, 6))\n",
        "    \n",
        "    plt.plot(train.index, train, label='Train', linewidth=2)\n",
        "    plt.plot(test.index, test, label='Test (Actual)', linewidth=2, color='orange')\n",
        "    plt.plot(test.index, prophet_forecast, label='Prophet Forecast', \n",
        "             linewidth=2, color='blue', linestyle='--')\n",
        "    \n",
        "    plt.axvline(train.index[-1], color='red', linestyle='--', alpha=0.5)\n",
        "    plt.title('Prophet Forecast', fontsize=16, fontweight='bold')\n",
        "    plt.xlabel('Time')\n",
        "    plt.ylabel('Passengers')\n",
        "    plt.legend()\n",
        "    plt.grid(alpha=0.3)\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "    \n",
        "    # Prophet компоненты\n",
        "    fig = prophet_model.plot_components(prophet_forecast_full)\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "    \n",
        "    print(\"\\n✅ Prophet автоматически разделил:\")\n",
        "    print(\"  - Тренд (общее направление)\")\n",
        "    print(\"  - Yearly seasonality (годовая сезонность)\")"
    ]
})

# ============================================================================
# COMPARISON
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.5 Сравнение всех моделей\n",
        "\n",
        "Сравним ARIMA, SARIMA и Prophet по метрикам и визуально."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Таблица сравнения\n",
        "comparison = pd.DataFrame({\n",
        "    'Model': ['ARIMA(1,1,1)', 'SARIMA(1,1,1)(1,1,1,12)', 'Prophet'],\n",
        "    'RMSE': [arima_rmse, sarima_rmse, prophet_rmse if prophet_available else None],\n",
        "    'MAE': [arima_mae, sarima_mae, prophet_mae if prophet_available else None],\n",
        "    'AIC': [arima_fitted.aic, sarima_fitted.aic, 'N/A'],\n",
        "    'Сезонность': ['❌', '✅', '✅'],\n",
        "    'Автоподбор': ['❌', '❌', '✅']\n",
        "})\n",
        "\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"СРАВНЕНИЕ МОДЕЛЕЙ\")\n",
        "print(\"=\"*80)\n",
        "print(comparison.to_string(index=False))\n",
        "\n",
        "# Визуальное сравнение\n",
        "plt.figure(figsize=(14, 7))\n",
        "\n",
        "plt.plot(train.index, train, label='Train', linewidth=2, alpha=0.7)\n",
        "plt.plot(test.index, test, label='Test (Actual)', linewidth=3, color='black')\n",
        "plt.plot(test.index, arima_forecast, label=f'ARIMA (RMSE={arima_rmse:.1f})', \n",
        "         linewidth=2, linestyle='--', alpha=0.8)\n",
        "plt.plot(test.index, sarima_forecast, label=f'SARIMA (RMSE={sarima_rmse:.1f})', \n",
        "         linewidth=2, linestyle='--', alpha=0.8)\n",
        "\n",
        "if prophet_available:\n",
        "    plt.plot(test.index, prophet_forecast, \n",
        "             label=f'Prophet (RMSE={prophet_rmse:.1f})', \n",
        "             linewidth=2, linestyle='--', alpha=0.8)\n",
        "\n",
        "plt.axvline(train.index[-1], color='red', linestyle='--', alpha=0.3, \n",
        "            linewidth=2, label='Train/Test Split')\n",
        "plt.title('Model Comparison: ARIMA vs SARIMA vs Prophet', \n",
        "          fontsize=16, fontweight='bold')\n",
        "plt.xlabel('Time', fontsize=12)\n",
        "plt.ylabel('Number of Passengers', fontsize=12)\n",
        "plt.legend(loc='upper left', fontsize=10)\n",
        "plt.grid(alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# ============================================================================
# RESIDUALS ANALYSIS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.6 Анализ остатков (Residuals Diagnostic)\n",
        "\n",
        "Хорошая модель должна иметь остатки близкие к **белому шуму**:\n",
        "- Нормальное распределение\n",
        "- Нулевое среднее\n",
        "- Постоянная дисперсия (гомоскедастичность)\n",
        "- Нет автокорреляции"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Диагностика остатков SARIMA (лучшая модель)\n",
        "fig = sarima_fitted.plot_diagnostics(figsize=(14, 10))\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n📊 Интерпретация диагностики остатков:\")\n",
        "print(\"\\n1. Standardized Residuals:\")\n",
        "print(\"   - Должны быть случайными (без паттернов)\")\n",
        "print(\"   - Среднее ≈ 0, дисперсия постоянная\")\n",
        "print(\"\\n2. Histogram + KDE:\")\n",
        "print(\"   - Остатки должны быть нормально распределены (гауссовская кривая)\")\n",
        "print(\"\\n3. Q-Q Plot:\")\n",
        "print(\"   - Точки должны лежать на диагональной линии\")\n",
        "print(\"   - Если да → остатки нормальные\")\n",
        "print(\"\\n4. Correlogram (ACF):\")\n",
        "print(\"   - Все лаги должны быть внутри доверительного интервала (синяя зона)\")\n",
        "print(\"   - Если да → нет автокорреляции (белый шум)\")"
    ]
})

# ============================================================================
# CONCLUSIONS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🎓 Выводы и рекомендации\n",
        "\n",
        "### 📊 Результаты\n",
        "\n",
        "**Производительность моделей:**\n",
        "1. **SARIMA** показала лучшие результаты (низкий RMSE)\n",
        "   - ✅ Учитывает сезонность\n",
        "   - ✅ Точнее ARIMA\n",
        "   - ⚠️ Требует подбора параметров\n",
        "\n",
        "2. **Prophet** близок к SARIMA\n",
        "   - ✅ Автоматический (не нужно подбирать p, d, q)\n",
        "   - ✅ Интерпретируемые компоненты\n",
        "   - ✅ Легко добавить праздники и внешние регрессоры\n",
        "   - ⚠️ Может быть менее точным на некоторых данных\n",
        "\n",
        "3. **ARIMA** без сезонности\n",
        "   - ❌ НЕ учитывает годовые циклы\n",
        "   - ❌ Худшая точность\n",
        "   - ✅ Проще для понимания\n",
        "\n",
        "---\n",
        "\n",
        "### 🎯 Когда использовать какую модель?\n",
        "\n",
        "| Модель | Когда использовать | Когда НЕ использовать |\n",
        "|--------|-------------------|----------------------|\n",
        "| **ARIMA** | - Нет четкой сезонности<br>- Короткие ряды<br>- Нужна простота | - Есть сезонность<br>- Нестационарный ряд |\n",
        "| **SARIMA** | - Четкая сезонность<br>- Средние/длинные ряды<br>- Нужна максимальная точность | - Нет сезонности<br>- Очень длинные ряды (медленно) |\n",
        "| **Prophet** | - Бизнес-метрики<br>- Праздники важны<br>- Нужна автоматизация<br>- Много пропусков | - Нужен полный контроль<br>- Нет явной сезонности |\n",
        "\n",
        "---\n",
        "\n",
        "### 🚀 Следующие шаги: Deep Learning для временных рядов\n",
        "\n",
        "**Классические методы (ARIMA/SARIMA/Prophet):**\n",
        "- ✅ Работают отлично на univariate временных рядах\n",
        "- ✅ Быстрые, интерпретируемые\n",
        "- ✅ Требуют мало данных\n",
        "- ❌ Сложно с multivariate (много признаков)\n",
        "- ❌ Линейные зависимости\n",
        "\n",
        "**Deep Learning (RNN/LSTM/GRU) - следующий ноутбук:**\n",
        "- ✅ Нелинейные зависимости\n",
        "- ✅ Multivariate временные ряды\n",
        "- ✅ Долгосрочные зависимости (LSTM)\n",
        "- ✅ Автоматическое извлечение признаков\n",
        "- ❌ Требуют много данных\n",
        "- ❌ Сложнее интерпретировать\n",
        "- ❌ Дольше обучаются\n",
        "\n",
        "**Рекомендация:**\n",
        "1. Начинайте с SARIMA/Prophet (быстро, хорошие результаты)\n",
        "2. Переходите к LSTM, если:\n",
        "   - Много признаков (multivariate)\n",
        "   - Нелинейные паттерны\n",
        "   - Большой датасет (1000+ точек)\n",
        "   - Нужна экстраполяция на длинные горизонты\n",
        "\n",
        "---\n",
        "\n",
        "### 📚 Дополнительные материалы\n",
        "\n",
        "**Теория:**\n",
        "- [\"Forecasting: Principles and Practice\" (Hyndman & Athanasopoulos)](https://otexts.com/fpp3/)\n",
        "- [\"Time Series Analysis\" (Hamilton)](https://press.princeton.edu/books/hardcover/9780691042893/time-series-analysis)\n",
        "\n",
        "**Практика:**\n",
        "- [statsmodels documentation](https://www.statsmodels.org/stable/tsa.html)\n",
        "- [Prophet documentation](https://facebook.github.io/prophet/)\n",
        "- [pmdarima (auto_arima)](http://alkaline-ml.com/pmdarima/)\n",
        "\n",
        "---\n",
        "\n",
        "**Phase 3, Step 1 COMPLETE!** ✅  \n",
        "**Next:** RNN/LSTM/GRU for Time Series (Step 2)"
    ]
})

# Сохраняем обновленный ноутбук
notebook['cells'] = cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Модели добавлены: {notebook_path}')
print(f'Всего ячеек: {len(cells)}')
print('Ноутбук Classical Time Series готов!')
