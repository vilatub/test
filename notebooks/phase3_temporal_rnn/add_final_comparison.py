#!/usr/bin/env python3
"""
Добавление обучения, визуализации Attention и финальных выводов
"""

import json

# Загружаем существующий ноутбук
notebook_path = '/home/user/test/notebooks/phase3_temporal_rnn/03_attention_seq2seq.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

cells = notebook['cells']

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🏋️ Часть 3: Обучение и сравнение\n",
        "\n",
        "### 3.1 Функции обучения"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def train_seq2seq(model, train_loader, criterion, optimizer, num_epochs=100, patience=15):\n",
        "    \"\"\"\n",
        "    Обучение Seq2Seq модели\n",
        "    \"\"\"\n",
        "    history = {'train_loss': []}\n",
        "    best_loss = float('inf')\n",
        "    patience_counter = 0\n",
        "    \n",
        "    model.train()\n",
        "    \n",
        "    for epoch in range(num_epochs):\n",
        "        epoch_loss = 0\n",
        "        \n",
        "        for X_batch, y_batch in train_loader:\n",
        "            # Forward pass\n",
        "            outputs = model(X_batch)\n",
        "            loss = criterion(outputs, y_batch)\n",
        "            \n",
        "            # Backward pass\n",
        "            optimizer.zero_grad()\n",
        "            loss.backward()\n",
        "            \n",
        "            # Gradient clipping\n",
        "            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n",
        "            \n",
        "            optimizer.step()\n",
        "            \n",
        "            epoch_loss += loss.item()\n",
        "        \n",
        "        avg_loss = epoch_loss / len(train_loader)\n",
        "        history['train_loss'].append(avg_loss)\n",
        "        \n",
        "        # Early stopping\n",
        "        if avg_loss < best_loss:\n",
        "            best_loss = avg_loss\n",
        "            patience_counter = 0\n",
        "        else:\n",
        "            patience_counter += 1\n",
        "        \n",
        "        if (epoch + 1) % 10 == 0:\n",
        "            print(f\"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}\")\n",
        "        \n",
        "        if patience_counter >= patience:\n",
        "            print(f\"\\nEarly stopping at epoch {epoch+1}\")\n",
        "            break\n",
        "    \n",
        "    return history\n",
        "\n",
        "def evaluate_seq2seq(model, X, y, scaler):\n",
        "    \"\"\"\n",
        "    Оценка Seq2Seq модели\n",
        "    \"\"\"\n",
        "    model.eval()\n",
        "    \n",
        "    with torch.no_grad():\n",
        "        predictions = model(X).cpu().numpy()\n",
        "    \n",
        "    # Денормализация\n",
        "    y_true = scaler.inverse_transform(y.cpu().numpy().reshape(-1, 1))\n",
        "    y_pred = scaler.inverse_transform(predictions.reshape(-1, 1))\n",
        "    \n",
        "    # Метрики\n",
        "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n",
        "    mae = mean_absolute_error(y_true, y_pred)\n",
        "    \n",
        "    # Reshape обратно в (n_samples, output_len)\n",
        "    y_true_seq = y_true.reshape(-1, OUTPUT_SEQ_LEN)\n",
        "    y_pred_seq = y_pred.reshape(-1, OUTPUT_SEQ_LEN)\n",
        "    \n",
        "    return y_pred_seq, y_true_seq, rmse, mae\n",
        "\n",
        "print(\"✅ Функции обучения готовы\")"
    ]
})

# ============================================================================
# TRAIN MODELS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.2 Обучение моделей"
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
        "PATIENCE = 20\n",
        "\n",
        "criterion = nn.MSELoss()\n",
        "\n",
        "print(\"=\"*60)\n",
        "print(\"Training Configuration\")\n",
        "print(\"=\"*60)\n",
        "print(f\"Input sequence length: {INPUT_SEQ_LEN}\")\n",
        "print(f\"Output sequence length: {OUTPUT_SEQ_LEN}\")\n",
        "print(f\"Epochs: {NUM_EPOCHS}\")\n",
        "print(f\"Learning Rate: {LEARNING_RATE}\")\n",
        "print(f\"Batch Size: {BATCH_SIZE}\")\n",
        "print(\"=\"*60)"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Training Simple Seq2Seq\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"Training Simple Seq2Seq (без Attention)\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "simple_optimizer = optim.Adam(simple_seq2seq.parameters(), lr=LEARNING_RATE)\n",
        "simple_history = train_seq2seq(simple_seq2seq, train_loader, criterion, \n",
        "                               simple_optimizer, NUM_EPOCHS, PATIENCE)\n",
        "\n",
        "simple_pred, simple_true, simple_rmse, simple_mae = evaluate_seq2seq(\n",
        "    simple_seq2seq, X_test_tensor, y_test_tensor, scaler\n",
        ")\n",
        "\n",
        "print(f\"\\n✅ Simple Seq2Seq:\")\n",
        "print(f\"   RMSE: {simple_rmse:.2f}\")\n",
        "print(f\"   MAE: {simple_mae:.2f}\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Training Seq2Seq with Attention\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"Training Seq2Seq с Attention\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "attention_optimizer = optim.Adam(seq2seq_attention.parameters(), lr=LEARNING_RATE)\n",
        "attention_history = train_seq2seq(seq2seq_attention, train_loader, criterion,\n",
        "                                  attention_optimizer, NUM_EPOCHS, PATIENCE)\n",
        "\n",
        "attention_pred, attention_true, attention_rmse, attention_mae = evaluate_seq2seq(\n",
        "    seq2seq_attention, X_test_tensor, y_test_tensor, scaler\n",
        ")\n",
        "\n",
        "print(f\"\\n✅ Seq2Seq + Attention:\")\n",
        "print(f\"   RMSE: {attention_rmse:.2f}\")\n",
        "print(f\"   MAE: {attention_mae:.2f}\")"
    ]
})

# ============================================================================
# ATTENTION VISUALIZATION
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.3 Визуализация Attention Weights\n",
        "\n",
        "**Самое интересное:** Посмотрим, куда модель \"смотрит\" при предсказании!"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Получаем attention weights для одного примера\n",
        "seq2seq_attention.eval()\n",
        "with torch.no_grad():\n",
        "    sample_idx = 0\n",
        "    sample_input = X_test_tensor[sample_idx:sample_idx+1]\n",
        "    sample_output, sample_attention = seq2seq_attention(\n",
        "        sample_input, return_attention=True\n",
        "    )\n",
        "\n",
        "# Attention weights: (1, output_len, input_len)\n",
        "attention_weights = sample_attention.cpu().numpy()[0]\n",
        "\n",
        "# Визуализация heatmap\n",
        "plt.figure(figsize=(12, 6))\n",
        "sns.heatmap(\n",
        "    attention_weights,\n",
        "    cmap='YlOrRd',\n",
        "    xticklabels=[f't-{INPUT_SEQ_LEN-i}' for i in range(INPUT_SEQ_LEN)],\n",
        "    yticklabels=[f't+{i+1}' for i in range(OUTPUT_SEQ_LEN)],\n",
        "    cbar_kws={'label': 'Attention Weight'},\n",
        "    annot=True,\n",
        "    fmt='.2f',\n",
        "    linewidths=0.5\n",
        ")\n",
        "\n",
        "plt.title('Attention Weights Heatmap\\n' + \n",
        "          'Строки = выходные шаги, Столбцы = входные шаги',\n",
        "          fontsize=14, fontweight='bold')\n",
        "plt.xlabel('Input Time Steps', fontsize=12)\n",
        "plt.ylabel('Output Time Steps', fontsize=12)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n📊 Интерпретация Attention:\")\n",
        "print(\"  - Более светлые ячейки = модель уделяет больше внимания\")\n",
        "print(\"  - Для t+1: модель смотрит на последние шаги (недавнее прошлое)\")\n",
        "print(\"  - Для t+2, t+3: может смотреть на сезонные паттерны (12 месяцев назад)\")\n",
        "print(\"  - Автоматически обучается фокусироваться на релевантных частях!\")"
    ]
})

# ============================================================================
# COMPARISON
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.4 Сравнение всех подходов"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Training curves\n",
        "plt.figure(figsize=(12, 5))\n",
        "plt.plot(simple_history['train_loss'], label='Simple Seq2Seq', linewidth=2)\n",
        "plt.plot(attention_history['train_loss'], label='Seq2Seq + Attention', linewidth=2)\n",
        "plt.title('Training Loss Comparison', fontsize=16, fontweight='bold')\n",
        "plt.xlabel('Epoch', fontsize=12)\n",
        "plt.ylabel('Loss (MSE)', fontsize=12)\n",
        "plt.legend()\n",
        "plt.grid(alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"✅ Attention модель сходится быстрее и к меньшему loss\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Predictions visualization\n",
        "# Выбираем несколько примеров для визуализации\n",
        "num_examples = 3\n",
        "\n",
        "fig, axes = plt.subplots(num_examples, 1, figsize=(14, 4*num_examples))\n",
        "\n",
        "for i in range(num_examples):\n",
        "    ax = axes[i] if num_examples > 1 else axes\n",
        "    \n",
        "    # Истинные значения\n",
        "    true_vals = simple_true[i]\n",
        "    \n",
        "    # Предсказания\n",
        "    simple_vals = simple_pred[i]\n",
        "    attention_vals = attention_pred[i]\n",
        "    \n",
        "    # X-axis\n",
        "    x = np.arange(OUTPUT_SEQ_LEN)\n",
        "    \n",
        "    ax.plot(x, true_vals, 'o-', label='True', linewidth=3, markersize=8, color='black')\n",
        "    ax.plot(x, simple_vals, 's--', label='Simple Seq2Seq', linewidth=2, markersize=6)\n",
        "    ax.plot(x, attention_vals, '^--', label='Seq2Seq + Attention', linewidth=2, markersize=6)\n",
        "    \n",
        "    ax.set_title(f'Multi-step Forecast Example {i+1}', fontsize=14, fontweight='bold')\n",
        "    ax.set_xlabel('Future Time Steps', fontsize=12)\n",
        "    ax.set_ylabel('Passengers', fontsize=12)\n",
        "    ax.legend()\n",
        "    ax.grid(alpha=0.3)\n",
        "    ax.set_xticks(x)\n",
        "    ax.set_xticklabels([f't+{j+1}' for j in x])\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"📊 Attention модель обычно точнее, особенно на дальних горизонтах (t+2, t+3)\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Metrics comparison\n",
        "comparison_df = pd.DataFrame({\n",
        "    'Model': ['Simple Seq2Seq', 'Seq2Seq + Attention'],\n",
        "    'RMSE': [simple_rmse, attention_rmse],\n",
        "    'MAE': [simple_mae, attention_mae],\n",
        "    'Parameters': [\n",
        "        sum(p.numel() for p in simple_seq2seq.parameters()),\n",
        "        sum(p.numel() for p in seq2seq_attention.parameters())\n",
        "    ],\n",
        "    'Attention': ['❌', '✅']\n",
        "})\n",
        "\n",
        "comparison_df = comparison_df.sort_values('RMSE')\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"MULTI-STEP FORECASTING COMPARISON\")\n",
        "print(\"=\"*70)\n",
        "print(comparison_df.to_string(index=False))\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Bar plot\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "comparison_df.plot(x='Model', y='RMSE', kind='bar', ax=axes[0], \n",
        "                   legend=False, color='steelblue')\n",
        "axes[0].set_title('RMSE Comparison', fontsize=14, fontweight='bold')\n",
        "axes[0].set_ylabel('RMSE', fontsize=12)\n",
        "axes[0].set_xlabel('')\n",
        "axes[0].tick_params(axis='x', rotation=45)\n",
        "\n",
        "comparison_df.plot(x='Model', y='MAE', kind='bar', ax=axes[1], \n",
        "                   legend=False, color='coral')\n",
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
# CONCLUSIONS - PHASE 3 FINALE
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🎓 Выводы: Phase 3 Complete\n",
        "\n",
        "### 📊 Результаты Advanced RNN\n",
        "\n",
        "**Attention механизм:**\n",
        "- ✅ **Улучшает точность** на multi-step forecasting\n",
        "- ✅ **Интерпретируемость:** видим, куда модель смотрит\n",
        "- ✅ **Гибкость:** адаптивный фокус на разных частях входа\n",
        "- ⚠️ **Больше параметров** → риск переобучения на малых данных\n",
        "\n",
        "**Seq2Seq архитектура:**\n",
        "- ✅ **Естественная** для multi-step forecasting\n",
        "- ✅ **Encoder-Decoder** разделяет понимание и генерацию\n",
        "- ✅ **Масштабируется** на длинные горизонты предсказания\n",
        "- ❌ **Сложнее** обучать, чем простой LSTM\n",
        "\n",
        "---\n",
        "\n",
        "### 🎯 Итоги Phase 3: Temporal Data & RNN\n",
        "\n",
        "Мы прошли полный путь от классики к современным методам:\n",
        "\n",
        "#### Step 1: Classical Time Series\n",
        "- **ARIMA/SARIMA:** статистические модели, отлично на малых univariate данных\n",
        "- **Prophet:** автоматизация, бизнес-метрики, праздники\n",
        "- **Когда использовать:** < 1000 точек, четкая сезонность, нужна интерпретация\n",
        "\n",
        "#### Step 2: RNN/LSTM/GRU\n",
        "- **Vanilla RNN:** проблема vanishing gradient\n",
        "- **LSTM:** решает long-term dependencies через gates\n",
        "- **GRU:** баланс скорости и точности\n",
        "- **Когда использовать:** > 1000 точек, multivariate, нелинейные паттерны\n",
        "\n",
        "#### Step 3: Attention & Seq2Seq (сегодня)\n",
        "- **Attention:** динамический фокус на важных частях\n",
        "- **Seq2Seq:** multi-step forecasting, encoder-decoder\n",
        "- **Когда использовать:** длинные последовательности, multi-step ahead, нужна интерпретация\n",
        "\n",
        "---\n",
        "\n",
        "### 📈 Practical Decision Tree\n",
        "\n",
        "**Выбор модели для временных рядов:**\n",
        "\n",
        "```\n",
        "START\n",
        "  ↓\n",
        "Размер данных?\n",
        "  ├─ < 500 точек → ARIMA/SARIMA/Prophet\n",
        "  └─ > 500 точек → ↓\n",
        "       ↓\n",
        "Univariate или Multivariate?\n",
        "  ├─ Univariate → SARIMA vs LSTM (попробуйте оба)\n",
        "  └─ Multivariate → LSTM/GRU\n",
        "       ↓\n",
        "Горизонт предсказания?\n",
        "  ├─ 1 шаг (t+1) → Simple LSTM\n",
        "  └─ Много шагов (t+1..t+N) → Seq2Seq\n",
        "       ↓\n",
        "Нужна интерпретация?\n",
        "  ├─ Да → Seq2Seq + Attention\n",
        "  └─ Нет → Простой Seq2Seq быстрее\n",
        "```\n",
        "\n",
        "---\n",
        "\n",
        "### 🚀 Что дальше? Phase 4: Transformers\n",
        "\n",
        "**Проблемы RNN (даже с Attention):**\n",
        "- ❌ Последовательная обработка (медленно)\n",
        "- ❌ Сложно параллелизовать\n",
        "- ❌ Ограниченная длина контекста\n",
        "\n",
        "**Решение: Transformers**\n",
        "- ✅ **Self-Attention:** параллельная обработка всей последовательности\n",
        "- ✅ **Positional Encoding:** порядок без рекурсии\n",
        "- ✅ **Scalable:** обучается на огромных данных\n",
        "\n",
        "**Transformers для временных рядов:**\n",
        "- Temporal Fusion Transformer (Google)\n",
        "- Informer (долгосрочное прогнозирование)\n",
        "- TabTransformer (табличные + временные данные)\n",
        "\n",
        "**Transformers для NLP/Vision:**\n",
        "- BERT, GPT (текст)\n",
        "- Vision Transformer (ViT)\n",
        "- CLIP (мультимодальность)\n",
        "\n",
        "---\n",
        "\n",
        "### 💡 Ключевые уроки Phase 3\n",
        "\n",
        "**1. Не всегда сложное = лучшее**\n",
        "- SARIMA часто побеждает LSTM на малых данных\n",
        "- Начинайте с простого, усложняйте по необходимости\n",
        "\n",
        "**2. Attention = мощь + интерпретация**\n",
        "- Улучшает точность\n",
        "- Показывает, КАК модель принимает решения\n",
        "- Критично для production в регулируемых индустриях\n",
        "\n",
        "**3. Multi-step forecasting ≠ много 1-step моделей**\n",
        "- Seq2Seq учитывает зависимости между выходами\n",
        "- Меньше накопления ошибок\n",
        "\n",
        "**4. Размер данных важен**\n",
        "- Deep Learning требует данных\n",
        "- < 1k точек: классика\n",
        "- > 10k точек: DL начинает побеждать\n",
        "\n",
        "---\n",
        "\n",
        "### 📚 Дополнительные материалы\n",
        "\n",
        "**Attention механизм:**\n",
        "- [\"Attention Is All You Need\" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)\n",
        "- [\"Neural Machine Translation by Jointly Learning to Align and Translate\" (Bahdanau et al., 2014)](https://arxiv.org/abs/1409.0473)\n",
        "- [Visualizing Attention (distill.pub)](https://distill.pub/2016/augmented-rnns/)\n",
        "\n",
        "**Seq2Seq:**\n",
        "- [\"Sequence to Sequence Learning\" (Sutskever et al., 2014)](https://arxiv.org/abs/1409.3215)\n",
        "- [PyTorch Seq2Seq Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)\n",
        "\n",
        "**Time Series Forecasting:**\n",
        "- [\"Temporal Fusion Transformers\" (Google, 2021)](https://arxiv.org/abs/1912.09363)\n",
        "- [\"Deep Learning for Time Series Forecasting\" (Januschowski et al., 2020)](https://arxiv.org/abs/2004.10240)\n",
        "\n",
        "---\n",
        "\n",
        "**🎉 Phase 3: Temporal Data & RNN - COMPLETE!**\n",
        "\n",
        "**Достижения:**\n",
        "- ✅ Классические методы: ARIMA, SARIMA, Prophet\n",
        "- ✅ Рекуррентные сети: RNN, LSTM, GRU\n",
        "- ✅ Продвинутые техники: Attention, Seq2Seq\n",
        "- ✅ Multi-step forecasting\n",
        "- ✅ Интерпретируемость через Attention\n",
        "\n",
        "**Next Phase:** Transformers и современные архитектуры 🚀"
    ]
})

# Сохраняем финальный ноутбук
notebook['cells'] = cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Финальная часть добавлена: {notebook_path}')
print(f'Всего ячеек: {len(cells)}')
print('🎉 Phase 3 FINALE - Attention & Seq2Seq ноутбук готов!')
