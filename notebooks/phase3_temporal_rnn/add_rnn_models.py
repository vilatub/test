#!/usr/bin/env python3
"""
Добавление моделей RNN, LSTM, GRU и практики в ноутбук
"""

import json

# Загружаем существующий ноутбук
notebook_path = '/home/user/test/notebooks/phase3_temporal_rnn/02_rnn_lstm_gru.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

cells = notebook['cells']

# ============================================================================
# DATA PREPARATION FOR RNN
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.3 Подготовка данных для RNN: Sliding Window\n",
        "\n",
        "**Задача:** Создать последовательности (X) и целевые значения (y).\n",
        "\n",
        "**Метод Sliding Window:**\n",
        "- Используем последние `seq_length` точек для предсказания следующей\n",
        "- Например: `[t-4, t-3, t-2, t-1]` → `t`\n",
        "\n",
        "**Пример:**\n",
        "```\n",
        "Данные: [10, 20, 30, 40, 50, 60]\n",
        "seq_length = 3\n",
        "\n",
        "X:           y:\n",
        "[10,20,30] → 40\n",
        "[20,30,40] → 50\n",
        "[30,40,50] → 60\n",
        "```"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def create_sequences(data, seq_length):\n",
        "    \"\"\"\n",
        "    Создает последовательности для RNN с помощью sliding window\n",
        "    \n",
        "    Args:\n",
        "        data: массив numpy shape (n_samples,)\n",
        "        seq_length: длина последовательности (look-back period)\n",
        "    \n",
        "    Returns:\n",
        "        X: shape (n_sequences, seq_length, 1)\n",
        "        y: shape (n_sequences, 1)\n",
        "    \"\"\"\n",
        "    X, y = [], []\n",
        "    \n",
        "    for i in range(len(data) - seq_length):\n",
        "        # Последовательность длины seq_length\n",
        "        sequence = data[i:i + seq_length]\n",
        "        # Следующее значение - целевое\n",
        "        target = data[i + seq_length]\n",
        "        \n",
        "        X.append(sequence)\n",
        "        y.append(target)\n",
        "    \n",
        "    X = np.array(X)\n",
        "    y = np.array(y)\n",
        "    \n",
        "    # Добавляем размерность признаков (для univariate = 1)\n",
        "    X = X.reshape(-1, seq_length, 1)\n",
        "    y = y.reshape(-1, 1)\n",
        "    \n",
        "    return X, y\n",
        "\n",
        "# Тест функции\n",
        "test_data = np.array([10, 20, 30, 40, 50, 60])\n",
        "test_X, test_y = create_sequences(test_data, seq_length=3)\n",
        "\n",
        "print(\"Пример sliding window:\")\n",
        "print(\"Original data:\", test_data)\n",
        "print(\"\\nSequences (X) and targets (y):\")\n",
        "for i in range(len(test_X)):\n",
        "    print(f\"X: {test_X[i].flatten()} → y: {test_y[i][0]}\")\n",
        "\n",
        "print(f\"\\nShape: X={test_X.shape}, y={test_y.shape}\")"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.4 Нормализация и Train/Test Split"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Извлекаем значения\n",
        "data = df['passengers'].values.astype(float)\n",
        "\n",
        "# Нормализация (важно для нейронных сетей!)\n",
        "scaler = MinMaxScaler(feature_range=(0, 1))\n",
        "data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()\n",
        "\n",
        "print(f\"Original range: [{data.min():.1f}, {data.max():.1f}]\")\n",
        "print(f\"Normalized range: [{data_normalized.min():.3f}, {data_normalized.max():.3f}]\")\n",
        "\n",
        "# Параметры\n",
        "SEQ_LENGTH = 12  # используем год данных для предсказания следующего месяца\n",
        "TRAIN_SIZE = 0.8\n",
        "\n",
        "# Train/Test split (по времени!)\n",
        "train_size = int(len(data_normalized) * TRAIN_SIZE)\n",
        "train_data = data_normalized[:train_size]\n",
        "test_data = data_normalized[train_size - SEQ_LENGTH:]  # включаем SEQ_LENGTH для первой последовательности\n",
        "\n",
        "print(f\"\\nTrain size: {len(train_data)} ({TRAIN_SIZE*100:.0f}%)\")\n",
        "print(f\"Test size: {len(test_data) - SEQ_LENGTH} (фактические предсказания)\")\n",
        "\n",
        "# Создаем последовательности\n",
        "X_train, y_train = create_sequences(train_data, SEQ_LENGTH)\n",
        "X_test, y_test = create_sequences(test_data, SEQ_LENGTH)\n",
        "\n",
        "print(f\"\\nTrain sequences: X={X_train.shape}, y={y_train.shape}\")\n",
        "print(f\"Test sequences: X={X_test.shape}, y={y_test.shape}\")\n",
        "\n",
        "# Преобразуем в PyTorch tensors\n",
        "X_train_tensor = torch.FloatTensor(X_train).to(device)\n",
        "y_train_tensor = torch.FloatTensor(y_train).to(device)\n",
        "X_test_tensor = torch.FloatTensor(X_test).to(device)\n",
        "y_test_tensor = torch.FloatTensor(y_test).to(device)\n",
        "\n",
        "# DataLoader\n",
        "BATCH_SIZE = 16\n",
        "\n",
        "train_dataset = TensorDataset(X_train_tensor, y_train_tensor)\n",
        "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)  # НЕ shuffle для TS!\n",
        "\n",
        "print(f\"\\nBatch size: {BATCH_SIZE}\")\n",
        "print(f\"Batches per epoch: {len(train_loader)}\")"
    ]
})

# ============================================================================
# VANILLA RNN MODEL
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.5 Модель 1: Vanilla RNN\n",
        "\n",
        "Простая рекуррентная сеть для базового сравнения."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class VanillaRNN(nn.Module):\n",
        "    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):\n",
        "        super(VanillaRNN, self).__init__()\n",
        "        \n",
        "        self.hidden_size = hidden_size\n",
        "        self.num_layers = num_layers\n",
        "        \n",
        "        # RNN layer\n",
        "        self.rnn = nn.RNN(\n",
        "            input_size=input_size,\n",
        "            hidden_size=hidden_size,\n",
        "            num_layers=num_layers,\n",
        "            batch_first=True  # (batch, seq, feature)\n",
        "        )\n",
        "        \n",
        "        # Fully connected layer\n",
        "        self.fc = nn.Linear(hidden_size, output_size)\n",
        "    \n",
        "    def forward(self, x):\n",
        "        # Инициализация hidden state\n",
        "        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)\n",
        "        \n",
        "        # RNN forward pass\n",
        "        # out: (batch, seq, hidden_size)\n",
        "        # h_n: (num_layers, batch, hidden_size)\n",
        "        out, h_n = self.rnn(x, h0)\n",
        "        \n",
        "        # Берем последний выход\n",
        "        out = out[:, -1, :]\n",
        "        \n",
        "        # Fully connected\n",
        "        out = self.fc(out)\n",
        "        \n",
        "        return out\n",
        "\n",
        "# Создание модели\n",
        "rnn_model = VanillaRNN(\n",
        "    input_size=1,\n",
        "    hidden_size=64,\n",
        "    num_layers=1,\n",
        "    output_size=1\n",
        ").to(device)\n",
        "\n",
        "print(\"Vanilla RNN Architecture:\")\n",
        "print(rnn_model)\n",
        "print(f\"\\nTotal parameters: {sum(p.numel() for p in rnn_model.parameters())}\")"
    ]
})

# ============================================================================
# LSTM MODEL
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.6 Модель 2: LSTM\n",
        "\n",
        "Long Short-Term Memory для долгосрочных зависимостей."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class LSTMModel(nn.Module):\n",
        "    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):\n",
        "        super(LSTMModel, self).__init__()\n",
        "        \n",
        "        self.hidden_size = hidden_size\n",
        "        self.num_layers = num_layers\n",
        "        \n",
        "        # LSTM layer\n",
        "        self.lstm = nn.LSTM(\n",
        "            input_size=input_size,\n",
        "            hidden_size=hidden_size,\n",
        "            num_layers=num_layers,\n",
        "            batch_first=True,\n",
        "            dropout=dropout if num_layers > 1 else 0  # dropout между слоями\n",
        "        )\n",
        "        \n",
        "        # Fully connected layers\n",
        "        self.fc = nn.Sequential(\n",
        "            nn.Linear(hidden_size, 32),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(dropout),\n",
        "            nn.Linear(32, output_size)\n",
        "        )\n",
        "    \n",
        "    def forward(self, x):\n",
        "        # Инициализация hidden и cell states\n",
        "        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)\n",
        "        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)\n",
        "        \n",
        "        # LSTM forward pass\n",
        "        out, (h_n, c_n) = self.lstm(x, (h0, c0))\n",
        "        \n",
        "        # Берем последний выход\n",
        "        out = out[:, -1, :]\n",
        "        \n",
        "        # Fully connected\n",
        "        out = self.fc(out)\n",
        "        \n",
        "        return out\n",
        "\n",
        "# Создание модели\n",
        "lstm_model = LSTMModel(\n",
        "    input_size=1,\n",
        "    hidden_size=64,\n",
        "    num_layers=2,\n",
        "    output_size=1,\n",
        "    dropout=0.2\n",
        ").to(device)\n",
        "\n",
        "print(\"LSTM Architecture:\")\n",
        "print(lstm_model)\n",
        "print(f\"\\nTotal parameters: {sum(p.numel() for p in lstm_model.parameters())}\")"
    ]
})

# ============================================================================
# GRU MODEL
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.7 Модель 3: GRU\n",
        "\n",
        "Gated Recurrent Unit - баланс между RNN и LSTM."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class GRUModel(nn.Module):\n",
        "    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):\n",
        "        super(GRUModel, self).__init__()\n",
        "        \n",
        "        self.hidden_size = hidden_size\n",
        "        self.num_layers = num_layers\n",
        "        \n",
        "        # GRU layer\n",
        "        self.gru = nn.GRU(\n",
        "            input_size=input_size,\n",
        "            hidden_size=hidden_size,\n",
        "            num_layers=num_layers,\n",
        "            batch_first=True,\n",
        "            dropout=dropout if num_layers > 1 else 0\n",
        "        )\n",
        "        \n",
        "        # Fully connected layers\n",
        "        self.fc = nn.Sequential(\n",
        "            nn.Linear(hidden_size, 32),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(dropout),\n",
        "            nn.Linear(32, output_size)\n",
        "        )\n",
        "    \n",
        "    def forward(self, x):\n",
        "        # Инициализация hidden state\n",
        "        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)\n",
        "        \n",
        "        # GRU forward pass\n",
        "        out, h_n = self.gru(x, h0)\n",
        "        \n",
        "        # Берем последний выход\n",
        "        out = out[:, -1, :]\n",
        "        \n",
        "        # Fully connected\n",
        "        out = self.fc(out)\n",
        "        \n",
        "        return out\n",
        "\n",
        "# Создание модели\n",
        "gru_model = GRUModel(\n",
        "    input_size=1,\n",
        "    hidden_size=64,\n",
        "    num_layers=2,\n",
        "    output_size=1,\n",
        "    dropout=0.2\n",
        ").to(device)\n",
        "\n",
        "print(\"GRU Architecture:\")\n",
        "print(gru_model)\n",
        "print(f\"\\nTotal parameters: {sum(p.numel() for p in gru_model.parameters())}\")"
    ]
})

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.8 Функции обучения и оценки"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def train_model(model, train_loader, criterion, optimizer, num_epochs=100, patience=10):\n",
        "    \"\"\"\n",
        "    Обучение RNN модели с early stopping\n",
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
        "            # Gradient clipping (предотвращает exploding gradients)\n",
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
        "def evaluate_model(model, X, y, scaler):\n",
        "    \"\"\"\n",
        "    Оценка модели и денормализация предсказаний\n",
        "    \"\"\"\n",
        "    model.eval()\n",
        "    \n",
        "    with torch.no_grad():\n",
        "        predictions = model(X).cpu().numpy()\n",
        "    \n",
        "    # Денормализация\n",
        "    y_true = scaler.inverse_transform(y.cpu().numpy())\n",
        "    y_pred = scaler.inverse_transform(predictions)\n",
        "    \n",
        "    # Метрики\n",
        "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n",
        "    mae = mean_absolute_error(y_true, y_pred)\n",
        "    \n",
        "    return y_pred, y_true, rmse, mae\n",
        "\n",
        "print(\"✅ Функции обучения готовы\")"
    ]
})

# Сохраняем обновленный ноутбук
notebook['cells'] = cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Модели добавлены: {notebook_path}')
print(f'Всего ячеек: {len(cells)}')
print('Следующая часть: обучение моделей и сравнение...')
