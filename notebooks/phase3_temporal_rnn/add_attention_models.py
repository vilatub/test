#!/usr/bin/env python3
"""
Добавление моделей с Attention и Seq2Seq в ноутбук
"""

import json

# Загружаем существующий ноутбук
notebook_path = '/home/user/test/notebooks/phase3_temporal_rnn/03_attention_seq2seq.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

cells = notebook['cells']

# ============================================================================
# DATA PREPARATION FOR SEQ2SEQ
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.3 Подготовка данных для Seq2Seq\n",
        "\n",
        "**Отличие от обычного LSTM:**\n",
        "- Вход: последовательность (например, 12 месяцев)\n",
        "- Выход: последовательность (например, 3 месяца)\n",
        "\n",
        "**Формат данных:**\n",
        "- `X`: shape (n_samples, input_seq_len, 1)\n",
        "- `y`: shape (n_samples, output_seq_len, 1)"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def create_seq2seq_data(data, input_len=12, output_len=3):\n",
        "    \"\"\"\n",
        "    Создает данные для Seq2Seq: последовательность → последовательность\n",
        "    \n",
        "    Args:\n",
        "        data: массив numpy\n",
        "        input_len: длина входной последовательности\n",
        "        output_len: длина выходной последовательности\n",
        "    \n",
        "    Returns:\n",
        "        X: shape (n_samples, input_len, 1)\n",
        "        y: shape (n_samples, output_len, 1)\n",
        "    \"\"\"\n",
        "    X, y = [], []\n",
        "    \n",
        "    for i in range(len(data) - input_len - output_len + 1):\n",
        "        # Входная последовательность\n",
        "        input_seq = data[i:i + input_len]\n",
        "        # Выходная последовательность (следующие output_len точек)\n",
        "        output_seq = data[i + input_len:i + input_len + output_len]\n",
        "        \n",
        "        X.append(input_seq)\n",
        "        y.append(output_seq)\n",
        "    \n",
        "    X = np.array(X).reshape(-1, input_len, 1)\n",
        "    y = np.array(y).reshape(-1, output_len, 1)\n",
        "    \n",
        "    return X, y\n",
        "\n",
        "# Параметры\n",
        "INPUT_SEQ_LEN = 12   # используем 12 месяцев\n",
        "OUTPUT_SEQ_LEN = 3   # предсказываем 3 месяца\n",
        "TRAIN_SIZE = 0.8\n",
        "\n",
        "# Нормализация\n",
        "data = df['passengers'].values.astype(float)\n",
        "scaler = MinMaxScaler(feature_range=(0, 1))\n",
        "data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()\n",
        "\n",
        "# Train/Test split\n",
        "train_size = int(len(data_normalized) * TRAIN_SIZE)\n",
        "train_data = data_normalized[:train_size]\n",
        "test_data = data_normalized[train_size - INPUT_SEQ_LEN:]  # overlap для первой последовательности\n",
        "\n",
        "# Создаем последовательности\n",
        "X_train, y_train = create_seq2seq_data(train_data, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)\n",
        "X_test, y_test = create_seq2seq_data(test_data, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)\n",
        "\n",
        "print(f\"Train: X={X_train.shape}, y={y_train.shape}\")\n",
        "print(f\"Test: X={X_test.shape}, y={y_test.shape}\")\n",
        "print(f\"\\nПример:\")\n",
        "print(f\"  Вход (12 месяцев): {X_train[0].flatten()[:5]}...\")\n",
        "print(f\"  Выход (3 месяца): {y_train[0].flatten()}\")\n",
        "\n",
        "# PyTorch tensors\n",
        "X_train_tensor = torch.FloatTensor(X_train).to(device)\n",
        "y_train_tensor = torch.FloatTensor(y_train).to(device)\n",
        "X_test_tensor = torch.FloatTensor(X_test).to(device)\n",
        "y_test_tensor = torch.FloatTensor(y_test).to(device)\n",
        "\n",
        "# DataLoader\n",
        "BATCH_SIZE = 8\n",
        "train_dataset = TensorDataset(X_train_tensor, y_train_tensor)\n",
        "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)"
    ]
})

# ============================================================================
# ATTENTION LAYER
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.4 Attention Layer\n",
        "\n",
        "Реализуем Bahdanau Attention (concat-based)."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class BahdanauAttention(nn.Module):\n",
        "    \"\"\"\n",
        "    Bahdanau Attention (также известен как Additive Attention)\n",
        "    \n",
        "    e_ti = v^T tanh(W_1 h_i + W_2 s_t)\n",
        "    alpha_ti = softmax(e_ti)\n",
        "    c_t = sum(alpha_ti * h_i)\n",
        "    \"\"\"\n",
        "    def __init__(self, hidden_size):\n",
        "        super(BahdanauAttention, self).__init__()\n",
        "        \n",
        "        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)\n",
        "        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)\n",
        "        self.v = nn.Linear(hidden_size, 1, bias=False)\n",
        "    \n",
        "    def forward(self, encoder_outputs, decoder_hidden):\n",
        "        \"\"\"\n",
        "        Args:\n",
        "            encoder_outputs: (batch, seq_len, hidden_size)\n",
        "            decoder_hidden: (batch, hidden_size)\n",
        "        \n",
        "        Returns:\n",
        "            context: (batch, hidden_size)\n",
        "            attention_weights: (batch, seq_len)\n",
        "        \"\"\"\n",
        "        # Расширяем decoder_hidden до (batch, seq_len, hidden_size)\n",
        "        seq_len = encoder_outputs.size(1)\n",
        "        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)\n",
        "        \n",
        "        # Compute attention scores\n",
        "        # e = v^T tanh(W1*h + W2*s)\n",
        "        energy = torch.tanh(\n",
        "            self.W1(encoder_outputs) + self.W2(decoder_hidden_expanded)\n",
        "        )\n",
        "        attention_scores = self.v(energy).squeeze(-1)  # (batch, seq_len)\n",
        "        \n",
        "        # Normalize with softmax\n",
        "        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len)\n",
        "        \n",
        "        # Compute context vector (weighted sum)\n",
        "        # c = sum(alpha_i * h_i)\n",
        "        context = torch.bmm(\n",
        "            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)\n",
        "            encoder_outputs  # (batch, seq_len, hidden_size)\n",
        "        ).squeeze(1)  # (batch, hidden_size)\n",
        "        \n",
        "        return context, attention_weights\n",
        "\n",
        "print(\"✅ BahdanauAttention реализован\")"
    ]
})

# ============================================================================
# BASELINE LSTM (для сравнения)
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.5 Baseline: Simple LSTM Seq2Seq (без Attention)\n",
        "\n",
        "Для сравнения создадим простой Seq2Seq без Attention."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class SimpleSeq2Seq(nn.Module):\n",
        "    \"\"\"\n",
        "    Базовый Seq2Seq без Attention:\n",
        "    - Encoder сжимает вход в один context vector\n",
        "    - Decoder генерирует последовательность из context\n",
        "    \"\"\"\n",
        "    def __init__(self, input_size=1, hidden_size=64, output_size=1, \n",
        "                 num_layers=2, output_len=3):\n",
        "        super(SimpleSeq2Seq, self).__init__()\n",
        "        \n",
        "        self.hidden_size = hidden_size\n",
        "        self.num_layers = num_layers\n",
        "        self.output_len = output_len\n",
        "        \n",
        "        # Encoder\n",
        "        self.encoder = nn.LSTM(\n",
        "            input_size, hidden_size, num_layers,\n",
        "            batch_first=True, dropout=0.2 if num_layers > 1 else 0\n",
        "        )\n",
        "        \n",
        "        # Decoder\n",
        "        self.decoder = nn.LSTM(\n",
        "            output_size, hidden_size, num_layers,\n",
        "            batch_first=True, dropout=0.2 if num_layers > 1 else 0\n",
        "        )\n",
        "        \n",
        "        # Output layer\n",
        "        self.fc = nn.Linear(hidden_size, output_size)\n",
        "    \n",
        "    def forward(self, x, target_len=None):\n",
        "        \"\"\"\n",
        "        Args:\n",
        "            x: (batch, seq_len, input_size)\n",
        "            target_len: int (если None, использует self.output_len)\n",
        "        \"\"\"\n",
        "        batch_size = x.size(0)\n",
        "        target_len = target_len or self.output_len\n",
        "        \n",
        "        # ENCODER\n",
        "        _, (hidden, cell) = self.encoder(x)\n",
        "        \n",
        "        # DECODER (autoregressive)\n",
        "        decoder_input = torch.zeros(batch_size, 1, 1).to(x.device)\n",
        "        outputs = []\n",
        "        \n",
        "        for _ in range(target_len):\n",
        "            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))\n",
        "            prediction = self.fc(decoder_output.squeeze(1))\n",
        "            outputs.append(prediction)\n",
        "            \n",
        "            # Следующий вход = текущее предсказание\n",
        "            decoder_input = prediction.unsqueeze(1)\n",
        "        \n",
        "        outputs = torch.stack(outputs, dim=1)  # (batch, target_len, 1)\n",
        "        return outputs\n",
        "\n",
        "# Создание модели\n",
        "simple_seq2seq = SimpleSeq2Seq(\n",
        "    input_size=1,\n",
        "    hidden_size=64,\n",
        "    output_size=1,\n",
        "    num_layers=2,\n",
        "    output_len=OUTPUT_SEQ_LEN\n",
        ").to(device)\n",
        "\n",
        "print(\"Simple Seq2Seq (без Attention):\")\n",
        "print(simple_seq2seq)\n",
        "print(f\"\\nПараметры: {sum(p.numel() for p in simple_seq2seq.parameters())}\")"
    ]
})

# ============================================================================
# SEQ2SEQ WITH ATTENTION
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.6 Seq2Seq с Attention\n",
        "\n",
        "Полная реализация Encoder-Decoder с Attention механизмом."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class Seq2SeqAttention(nn.Module):\n",
        "    \"\"\"\n",
        "    Seq2Seq с Bahdanau Attention:\n",
        "    - Encoder создает последовательность скрытых состояний\n",
        "    - Decoder на каждом шаге использует Attention для фокуса\n",
        "    - Context vector комбинируется с decoder state\n",
        "    \"\"\"\n",
        "    def __init__(self, input_size=1, hidden_size=64, output_size=1,\n",
        "                 num_layers=2, output_len=3):\n",
        "        super(Seq2SeqAttention, self).__init__()\n",
        "        \n",
        "        self.hidden_size = hidden_size\n",
        "        self.num_layers = num_layers\n",
        "        self.output_len = output_len\n",
        "        \n",
        "        # Encoder\n",
        "        self.encoder = nn.LSTM(\n",
        "            input_size, hidden_size, num_layers,\n",
        "            batch_first=True, dropout=0.2 if num_layers > 1 else 0\n",
        "        )\n",
        "        \n",
        "        # Attention\n",
        "        self.attention = BahdanauAttention(hidden_size)\n",
        "        \n",
        "        # Decoder (input = предыдущее предсказание + context)\n",
        "        self.decoder = nn.LSTM(\n",
        "            output_size + hidden_size,  # concatenate input and context\n",
        "            hidden_size, num_layers,\n",
        "            batch_first=True, dropout=0.2 if num_layers > 1 else 0\n",
        "        )\n",
        "        \n",
        "        # Output layer (decoder hidden + context)\n",
        "        self.fc = nn.Sequential(\n",
        "            nn.Linear(hidden_size * 2, hidden_size),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(hidden_size, output_size)\n",
        "        )\n",
        "    \n",
        "    def forward(self, x, target_len=None, return_attention=False):\n",
        "        \"\"\"\n",
        "        Args:\n",
        "            x: (batch, seq_len, input_size)\n",
        "            target_len: int\n",
        "            return_attention: bool (возвращать ли attention weights)\n",
        "        \n",
        "        Returns:\n",
        "            outputs: (batch, target_len, output_size)\n",
        "            attention_weights: (batch, target_len, seq_len) если return_attention=True\n",
        "        \"\"\"\n",
        "        batch_size = x.size(0)\n",
        "        target_len = target_len or self.output_len\n",
        "        \n",
        "        # ENCODER: получаем ВСЕ скрытые состояния\n",
        "        encoder_outputs, (hidden, cell) = self.encoder(x)\n",
        "        # encoder_outputs: (batch, seq_len, hidden_size)\n",
        "        \n",
        "        # DECODER с ATTENTION\n",
        "        decoder_input = torch.zeros(batch_size, 1, 1).to(x.device)\n",
        "        outputs = []\n",
        "        attention_weights_list = []\n",
        "        \n",
        "        for t in range(target_len):\n",
        "            # 1. Attention: вычисляем context vector\n",
        "            decoder_hidden = hidden[-1]  # последний слой\n",
        "            context, attention_weights = self.attention(encoder_outputs, decoder_hidden)\n",
        "            \n",
        "            # 2. Concatenate decoder input and context\n",
        "            decoder_input_combined = torch.cat(\n",
        "                [decoder_input, context.unsqueeze(1)], dim=2\n",
        "            )  # (batch, 1, output_size + hidden_size)\n",
        "            \n",
        "            # 3. Decoder step\n",
        "            decoder_output, (hidden, cell) = self.decoder(\n",
        "                decoder_input_combined, (hidden, cell)\n",
        "            )\n",
        "            \n",
        "            # 4. Output prediction (decoder hidden + context)\n",
        "            combined = torch.cat([decoder_output.squeeze(1), context], dim=1)\n",
        "            prediction = self.fc(combined)\n",
        "            \n",
        "            outputs.append(prediction)\n",
        "            attention_weights_list.append(attention_weights)\n",
        "            \n",
        "            # 5. Следующий вход = текущее предсказание\n",
        "            decoder_input = prediction.unsqueeze(1)\n",
        "        \n",
        "        outputs = torch.stack(outputs, dim=1)  # (batch, target_len, output_size)\n",
        "        \n",
        "        if return_attention:\n",
        "            attention_weights_tensor = torch.stack(attention_weights_list, dim=1)\n",
        "            return outputs, attention_weights_tensor\n",
        "        \n",
        "        return outputs\n",
        "\n",
        "# Создание модели\n",
        "seq2seq_attention = Seq2SeqAttention(\n",
        "    input_size=1,\n",
        "    hidden_size=64,\n",
        "    output_size=1,\n",
        "    num_layers=2,\n",
        "    output_len=OUTPUT_SEQ_LEN\n",
        ").to(device)\n",
        "\n",
        "print(\"Seq2Seq с Bahdanau Attention:\")\n",
        "print(seq2seq_attention)\n",
        "print(f\"\\nПараметры: {sum(p.numel() for p in seq2seq_attention.parameters())}\")"
    ]
})

# Сохраняем обновленный ноутбук
notebook['cells'] = cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Модели Attention и Seq2Seq добавлены: {notebook_path}')
print(f'Всего ячеек: {len(cells)}')
print('Следующая часть: обучение и сравнение...')
