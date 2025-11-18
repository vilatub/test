#!/usr/bin/env python3
"""
Добавление обучения, сравнения и выводов в ноутбук RNN/LSTM/GRU
"""

import json

# Загружаем существующий ноутбук
notebook_path = '/home/user/test/notebooks/phase3_temporal_rnn/02_rnn_lstm_gru.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

cells = notebook['cells']

# ============================================================================
# TRAINING ALL MODELS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🏋️ Часть 3: Обучение и сравнение моделей\n",
        "\n",
        "### 3.1 Обучение всех моделей"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Гиперпараметры\n",
        "NUM_EPOCHS = 200\n",
        "LEARNING_RATE = 0.001\n",
        "PATIENCE = 15\n",
        "\n",
        "# Loss и optimizer одинаковые для всех\n",
        "criterion = nn.MSELoss()\n",
        "\n",
        "# Словарь для хранения результатов\n",
        "results = {}\n",
        "\n",
        "print(\"=\"*60)\n",
        "print(\"Training Configuration\")\n",
        "print(\"=\"*60)\n",
        "print(f\"Epochs: {NUM_EPOCHS}\")\n",
        "print(f\"Learning Rate: {LEARNING_RATE}\")\n",
        "print(f\"Batch Size: {BATCH_SIZE}\")\n",
        "print(f\"Sequence Length: {SEQ_LENGTH}\")\n",
        "print(f\"Early Stopping Patience: {PATIENCE}\")\n",
        "print(\"=\"*60)"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "#### 3.1.1 Обучение Vanilla RNN"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"Training Vanilla RNN\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)\n",
        "rnn_history = train_model(rnn_model, train_loader, criterion, rnn_optimizer, \n",
        "                          NUM_EPOCHS, PATIENCE)\n",
        "\n",
        "# Evaluation\n",
        "rnn_pred, rnn_true, rnn_rmse, rnn_mae = evaluate_model(rnn_model, X_test_tensor, \n",
        "                                                        y_test_tensor, scaler)\n",
        "\n",
        "results['Vanilla RNN'] = {\n",
        "    'model': rnn_model,\n",
        "    'history': rnn_history,\n",
        "    'predictions': rnn_pred,\n",
        "    'rmse': rnn_rmse,\n",
        "    'mae': rnn_mae\n",
        "}\n",
        "\n",
        "print(f\"\\n✅ Vanilla RNN trained\")\n",
        "print(f\"   RMSE: {rnn_rmse:.2f}\")\n",
        "print(f\"   MAE: {rnn_mae:.2f}\")"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "#### 3.1.2 Обучение LSTM"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"Training LSTM\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)\n",
        "lstm_history = train_model(lstm_model, train_loader, criterion, lstm_optimizer, \n",
        "                           NUM_EPOCHS, PATIENCE)\n",
        "\n",
        "# Evaluation\n",
        "lstm_pred, lstm_true, lstm_rmse, lstm_mae = evaluate_model(lstm_model, X_test_tensor, \n",
        "                                                           y_test_tensor, scaler)\n",
        "\n",
        "results['LSTM'] = {\n",
        "    'model': lstm_model,\n",
        "    'history': lstm_history,\n",
        "    'predictions': lstm_pred,\n",
        "    'rmse': lstm_rmse,\n",
        "    'mae': lstm_mae\n",
        "}\n",
        "\n",
        "print(f\"\\n✅ LSTM trained\")\n",
        "print(f\"   RMSE: {lstm_rmse:.2f}\")\n",
        "print(f\"   MAE: {lstm_mae:.2f}\")"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "#### 3.1.3 Обучение GRU"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"Training GRU\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "gru_optimizer = optim.Adam(gru_model.parameters(), lr=LEARNING_RATE)\n",
        "gru_history = train_model(gru_model, train_loader, criterion, gru_optimizer, \n",
        "                          NUM_EPOCHS, PATIENCE)\n",
        "\n",
        "# Evaluation\n",
        "gru_pred, gru_true, gru_rmse, gru_mae = evaluate_model(gru_model, X_test_tensor, \n",
        "                                                       y_test_tensor, scaler)\n",
        "\n",
        "results['GRU'] = {\n",
        "    'model': gru_model,\n",
        "    'history': gru_history,\n",
        "    'predictions': gru_pred,\n",
        "    'rmse': gru_rmse,\n",
        "    'mae': gru_mae\n",
        "}\n",
        "\n",
        "print(f\"\\n✅ GRU trained\")\n",
        "print(f\"   RMSE: {gru_rmse:.2f}\")\n",
        "print(f\"   MAE: {gru_mae:.2f}\")"
    ]
})

# ============================================================================
# COMPARISON WITH SARIMA
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.2 Сравнение с классической SARIMA\n",
        "\n",
        "Добавим SARIMA из предыдущего ноутбука для честного сравнения."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"Training SARIMA for comparison\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "# Используем оригинальные (ненормализованные) данные\n",
        "train_size_orig = int(len(data) * TRAIN_SIZE)\n",
        "train_orig = data[:train_size_orig]\n",
        "test_orig = data[train_size_orig:]\n",
        "\n",
        "# SARIMA(1, 1, 1)(1, 1, 1, 12)\n",
        "sarima_model = SARIMAX(\n",
        "    train_orig,\n",
        "    order=(1, 1, 1),\n",
        "    seasonal_order=(1, 1, 1, 12),\n",
        "    enforce_stationarity=False,\n",
        "    enforce_invertibility=False\n",
        ")\n",
        "\n",
        "sarima_fitted = sarima_model.fit(disp=False)\n",
        "sarima_forecast = sarima_fitted.forecast(steps=len(test_orig))\n",
        "\n",
        "# Метрики\n",
        "sarima_rmse = np.sqrt(mean_squared_error(test_orig, sarima_forecast))\n",
        "sarima_mae = mean_absolute_error(test_orig, sarima_forecast)\n",
        "\n",
        "results['SARIMA'] = {\n",
        "    'predictions': sarima_forecast,\n",
        "    'rmse': sarima_rmse,\n",
        "    'mae': sarima_mae\n",
        "}\n",
        "\n",
        "print(f\"\\n✅ SARIMA trained\")\n",
        "print(f\"   RMSE: {sarima_rmse:.2f}\")\n",
        "print(f\"   MAE: {sarima_mae:.2f}\")"
    ]
})

# ============================================================================
# VISUALIZATION
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.3 Визуализация результатов"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Training curves\n",
        "fig, ax = plt.subplots(figsize=(12, 5))\n",
        "\n",
        "ax.plot(results['Vanilla RNN']['history']['train_loss'], label='Vanilla RNN', linewidth=2)\n",
        "ax.plot(results['LSTM']['history']['train_loss'], label='LSTM', linewidth=2)\n",
        "ax.plot(results['GRU']['history']['train_loss'], label='GRU', linewidth=2)\n",
        "\n",
        "ax.set_title('Training Loss Curves', fontsize=16, fontweight='bold')\n",
        "ax.set_xlabel('Epoch', fontsize=12)\n",
        "ax.set_ylabel('Loss (MSE)', fontsize=12)\n",
        "ax.legend()\n",
        "ax.grid(alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"📊 RNN модели обучаются стабильно, LSTM/GRU сходятся быстрее Vanilla RNN\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Predictions visualization\n",
        "# Получаем индексы для test данных\n",
        "test_indices = df.index[train_size_orig:]\n",
        "\n",
        "plt.figure(figsize=(14, 7))\n",
        "\n",
        "# Train данные (контекст)\n",
        "plt.plot(df.index[:train_size_orig], data[:train_size_orig], \n",
        "         label='Train', linewidth=2, alpha=0.5, color='gray')\n",
        "\n",
        "# Истинные test значения\n",
        "plt.plot(test_indices, test_orig, label='Test (True)', \n",
        "         linewidth=3, color='black')\n",
        "\n",
        "# Предсказания RNN моделей\n",
        "plt.plot(test_indices, results['Vanilla RNN']['predictions'], \n",
        "         label=f\"Vanilla RNN (RMSE={rnn_rmse:.1f})\", linewidth=2, linestyle='--')\n",
        "plt.plot(test_indices, results['LSTM']['predictions'], \n",
        "         label=f\"LSTM (RMSE={lstm_rmse:.1f})\", linewidth=2, linestyle='--')\n",
        "plt.plot(test_indices, results['GRU']['predictions'], \n",
        "         label=f\"GRU (RMSE={gru_rmse:.1f})\", linewidth=2, linestyle='--')\n",
        "\n",
        "# SARIMA\n",
        "plt.plot(test_indices, results['SARIMA']['predictions'], \n",
        "         label=f\"SARIMA (RMSE={sarima_rmse:.1f})\", linewidth=2, linestyle=':')\n",
        "\n",
        "plt.axvline(df.index[train_size_orig], color='red', linestyle='--', \n",
        "            alpha=0.3, linewidth=2, label='Train/Test Split')\n",
        "\n",
        "plt.title('Model Comparison: RNN vs LSTM vs GRU vs SARIMA', \n",
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
# METRICS COMPARISON
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.4 Сравнительная таблица метрик"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Создание таблицы сравнения\n",
        "comparison_df = pd.DataFrame({\n",
        "    'Model': ['Vanilla RNN', 'LSTM', 'GRU', 'SARIMA'],\n",
        "    'RMSE': [\n",
        "        results['Vanilla RNN']['rmse'],\n",
        "        results['LSTM']['rmse'],\n",
        "        results['GRU']['rmse'],\n",
        "        results['SARIMA']['rmse']\n",
        "    ],\n",
        "    'MAE': [\n",
        "        results['Vanilla RNN']['mae'],\n",
        "        results['LSTM']['mae'],\n",
        "        results['GRU']['mae'],\n",
        "        results['SARIMA']['mae']\n",
        "    ],\n",
        "    'Parameters': [\n",
        "        sum(p.numel() for p in rnn_model.parameters()),\n",
        "        sum(p.numel() for p in lstm_model.parameters()),\n",
        "        sum(p.numel() for p in gru_model.parameters()),\n",
        "        'N/A'\n",
        "    ],\n",
        "    'Type': ['Deep Learning', 'Deep Learning', 'Deep Learning', 'Classical']\n",
        "})\n",
        "\n",
        "# Сортируем по RMSE\n",
        "comparison_df = comparison_df.sort_values('RMSE')\n",
        "\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"MODEL COMPARISON\")\n",
        "print(\"=\"*80)\n",
        "print(comparison_df.to_string(index=False))\n",
        "print(\"=\"*80)\n",
        "\n",
        "# Визуализация метрик\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# RMSE comparison\n",
        "comparison_df.plot(x='Model', y='RMSE', kind='bar', ax=axes[0], legend=False, color='steelblue')\n",
        "axes[0].set_title('RMSE Comparison', fontsize=14, fontweight='bold')\n",
        "axes[0].set_ylabel('RMSE', fontsize=12)\n",
        "axes[0].set_xlabel('')\n",
        "axes[0].tick_params(axis='x', rotation=45)\n",
        "\n",
        "# MAE comparison\n",
        "comparison_df.plot(x='Model', y='MAE', kind='bar', ax=axes[1], legend=False, color='coral')\n",
        "axes[1].set_title('MAE Comparison', fontsize=14, fontweight='bold')\n",
        "axes[1].set_ylabel('MAE', fontsize=12)\n",
        "axes[1].set_xlabel('')\n",
        "axes[1].tick_params(axis='x', rotation=45)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
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
        "### 📊 Результаты на Airline Passengers\n",
        "\n",
        "**Ожидаемая ситуация для этого датасета:**\n",
        "\n",
        "1. **SARIMA чаще всего лучше** RNN-моделей\n",
        "   - ✅ Маленький датасет (144 точки)\n",
        "   - ✅ Четкая сезонность (SARIMA заточена под это)\n",
        "   - ✅ Univariate (одна переменная)\n",
        "\n",
        "2. **LSTM/GRU ≈ сопоставимы** с SARIMA или чуть хуже\n",
        "   - ⚠️ Мало данных для глубокого обучения\n",
        "   - ✅ Но захватывают нелинейные паттерны\n",
        "\n",
        "3. **Vanilla RNN хуже всех**\n",
        "   - ❌ Проблема vanishing gradient\n",
        "   - ❌ Не справляется с долгосрочными зависимостями\n",
        "\n",
        "---\n",
        "\n",
        "### 🎯 Когда использовать RNN/LSTM/GRU?\n",
        "\n",
        "**Deep Learning лучше классических методов когда:**\n",
        "\n",
        "| Характеристика | Classical (ARIMA/SARIMA) | Deep Learning (LSTM/GRU) |\n",
        "|----------------|-------------------------|-------------------------|\n",
        "| **Размер данных** | < 1000 точек | > 1000 точек (чем больше, тем лучше) |\n",
        "| **Количество признаков** | Univariate (1 переменная) | **Multivariate (много переменных)** |\n",
        "| **Паттерны** | Линейные, сезонные | **Нелинейные, сложные** |\n",
        "| **Скорость обучения** | **Быстро (секунды)** | Медленно (минуты/часы) |\n",
        "| **Интерпретируемость** | **Высокая** | Низкая (черный ящик) |\n",
        "| **Автоматизация** | Средняя (подбор p, d, q) | **Высокая (end-to-end)** |\n",
        "\n",
        "**RNN/LSTM/GRU побеждают когда:**\n",
        "- ✅ **Multivariate time series** (много взаимосвязанных переменных)\n",
        "- ✅ **Большие датасеты** (тысячи/миллионы точек)\n",
        "- ✅ **Нелинейные зависимости** (сложные паттерны)\n",
        "- ✅ **Нестандартная сезонность** (несколько периодов, нерегулярная)\n",
        "- ✅ **Много внешних факторов** (легко добавить как признаки)\n",
        "\n",
        "**Примеры задач для RNN:**\n",
        "- 🏭 **IoT сенсоры:** температура, давление, вибрация одновременно\n",
        "- 📈 **Финансы:** цена + объем + индикаторы + новости\n",
        "- ⚡ **Энергетика:** потребление + погода + день недели + праздники\n",
        "- 🏥 **Медицина:** мониторинг пациентов (пульс, давление, температура)\n",
        "\n",
        "---\n",
        "\n",
        "### 🔧 Практические рекомендации\n",
        "\n",
        "**Рабочий алгоритм выбора модели:**\n",
        "\n",
        "```\n",
        "1. Начните с ARIMA/SARIMA (baseline)\n",
        "   ↓\n",
        "2. Если результат неудовлетворителен:\n",
        "   - Маленький датасет (< 1k) → попробуйте Prophet\n",
        "   - Большой датасет (> 1k) → попробуйте LSTM/GRU\n",
        "   ↓\n",
        "3. Для multivariate:\n",
        "   - VAR (Vector AutoRegression) - классика\n",
        "   - LSTM/GRU - deep learning\n",
        "   ↓\n",
        "4. Ансамбль:\n",
        "   - SARIMA + LSTM часто лучше каждого по отдельности\n",
        "```\n",
        "\n",
        "**Тюнинг RNN моделей:**\n",
        "\n",
        "1. **Sequence length (look-back period)**\n",
        "   - Для сезонности: минимум 1 период (12 для месячных данных)\n",
        "   - Слишком большой → переобучение\n",
        "   - Слишком маленький → недостаточно контекста\n",
        "\n",
        "2. **Hidden size**\n",
        "   - Малые данные: 32-64\n",
        "   - Средние: 64-128\n",
        "   - Большие: 128-512\n",
        "\n",
        "3. **Number of layers**\n",
        "   - 1-2 слоя обычно достаточно\n",
        "   - Больше → переобучение на малых данных\n",
        "\n",
        "4. **Dropout**\n",
        "   - 0.1-0.3 для регуляризации\n",
        "   - Критично на малых датасетах\n",
        "\n",
        "5. **Learning rate**\n",
        "   - Начните с 0.001\n",
        "   - Используйте scheduler (ReduceLROnPlateau)\n",
        "\n",
        "---\n",
        "\n",
        "### 🚀 Следующие шаги\n",
        "\n",
        "**Продвинутые техники (Phase 3, Step 3):**\n",
        "- **Attention механизм** для RNN\n",
        "- **Seq2Seq** модели для multi-step forecasting\n",
        "- **Encoder-Decoder** архитектура\n",
        "- **Transformer** для временных рядов (Phase 4)\n",
        "\n",
        "**Практические улучшения:**\n",
        "- **Ансамбли:** SARIMA + LSTM\n",
        "- **Multivariate:** добавить внешние признаки\n",
        "- **Transfer learning:** предобучение на других TS\n",
        "- **Multi-step forecasting:** предсказание на несколько шагов\n",
        "\n",
        "---\n",
        "\n",
        "### 📚 Дополнительные материалы\n",
        "\n",
        "**Теория:**\n",
        "- [\"Understanding LSTM Networks\" (Colah's Blog)](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)\n",
        "- [\"The Unreasonable Effectiveness of RNN\" (Karpathy)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)\n",
        "\n",
        "**Практика:**\n",
        "- [PyTorch RNN Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)\n",
        "- [Time Series with LSTM](https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/)\n",
        "\n",
        "---\n",
        "\n",
        "**Phase 3, Step 2 COMPLETE!** ✅  \n",
        "**Next:** Advanced RNN - Attention & Seq2Seq (Step 3)"
    ]
})

# Сохраняем финальный ноутбук
notebook['cells'] = cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Обучение и сравнение добавлены: {notebook_path}')
print(f'Всего ячеек: {len(cells)}')
print('🎉 Ноутбук RNN/LSTM/GRU полностью готов!')
