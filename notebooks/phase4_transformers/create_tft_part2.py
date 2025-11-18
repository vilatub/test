#!/usr/bin/env python3
"""
Phase 4 Step 3: Temporal Fusion Transformer
Part 2: Data Preparation, TFT Building Blocks
"""

import json

# Загружаем существующий notebook
notebook_path = '/home/user/test/notebooks/phase4_transformers/03_temporal_fusion_transformer.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

cells = notebook['cells']

# ============================================================================
# DATA PREPARATION
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## 🔧 Часть 2: Data Preparation для TFT\n",
        "\n",
        "### 2.1 Sliding Window Approach\n",
        "\n",
        "**TFT требует:**\n",
        "- **encoder_length**: сколько historical timesteps использовать (например, 168 = 7 days)\n",
        "- **decoder_length**: сколько future timesteps предсказать (например, 24 = 1 day)\n",
        "\n",
        "**Пример:**\n",
        "```\n",
        "encoder_length = 168 (7 days × 24 hours)\n",
        "decoder_length = 24  (1 day)\n",
        "\n",
        "Historical: [t-168, ..., t-1]  → predict Future: [t, t+1, ..., t+23]\n",
        "```"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Hyperparameters\n",
        "ENCODER_LENGTH = 168  # 7 days lookback\n",
        "DECODER_LENGTH = 24   # 1 day prediction\n",
        "\n",
        "print(f\"TFT Window Configuration:\")\n",
        "print(f\"  Encoder length: {ENCODER_LENGTH} hours ({ENCODER_LENGTH//24} days)\")\n",
        "print(f\"  Decoder length: {DECODER_LENGTH} hours ({DECODER_LENGTH//24} day)\")\n",
        "print(f\"  Total window: {ENCODER_LENGTH + DECODER_LENGTH} hours\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def create_tft_dataset(df, encoder_length, decoder_length):\n",
        "    \"\"\"\n",
        "    Create TFT dataset with sliding windows\n",
        "    \n",
        "    Returns:\n",
        "        sequences: list of dicts with encoder/decoder data\n",
        "    \"\"\"\n",
        "    sequences = []\n",
        "    \n",
        "    # Process each household separately\n",
        "    for household_id in df['household_id'].unique():\n",
        "        household_df = df[df['household_id'] == household_id].sort_values('timestamp').reset_index(drop=True)\n",
        "        \n",
        "        total_length = encoder_length + decoder_length\n",
        "        \n",
        "        for i in range(len(household_df) - total_length + 1):\n",
        "            # Encoder data (historical)\n",
        "            encoder_df = household_df.iloc[i:i+encoder_length]\n",
        "            \n",
        "            # Decoder data (future to predict)\n",
        "            decoder_df = household_df.iloc[i+encoder_length:i+total_length]\n",
        "            \n",
        "            sequence = {\n",
        "                # Static features (same for all timesteps)\n",
        "                'household_id': household_id,\n",
        "                \n",
        "                # Encoder inputs (historical)\n",
        "                'encoder_hour': encoder_df['hour'].values,\n",
        "                'encoder_dow': encoder_df['day_of_week'].values,\n",
        "                'encoder_weekend': encoder_df['is_weekend'].values,\n",
        "                'encoder_temp': encoder_df['temperature'].values,\n",
        "                'encoder_consumption': encoder_df['consumption'].values,\n",
        "                \n",
        "                # Decoder inputs (future known)\n",
        "                'decoder_hour': decoder_df['hour'].values,\n",
        "                'decoder_dow': decoder_df['day_of_week'].values,\n",
        "                'decoder_weekend': decoder_df['is_weekend'].values,\n",
        "                \n",
        "                # Target (what to predict)\n",
        "                'target': decoder_df['consumption'].values\n",
        "            }\n",
        "            \n",
        "            sequences.append(sequence)\n",
        "    \n",
        "    return sequences\n",
        "\n",
        "print(\"Creating TFT dataset...\")\n",
        "sequences = create_tft_dataset(df, ENCODER_LENGTH, DECODER_LENGTH)\n",
        "\n",
        "print(f\"\\n✅ Created {len(sequences):,} sequences\")\n",
        "print(f\"\\nExample sequence keys: {list(sequences[0].keys())}\")\n",
        "print(f\"\\nExample shapes:\")\n",
        "for key, val in sequences[0].items():\n",
        "    if isinstance(val, np.ndarray):\n",
        "        print(f\"  {key:25s}: {val.shape}\")\n",
        "    else:\n",
        "        print(f\"  {key:25s}: {val}\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Train/Validation/Test Split\n",
        "# Split by time (not random!) for time series\n",
        "\n",
        "n_sequences = len(sequences)\n",
        "train_size = int(0.7 * n_sequences)\n",
        "val_size = int(0.15 * n_sequences)\n",
        "\n",
        "train_sequences = sequences[:train_size]\n",
        "val_sequences = sequences[train_size:train_size+val_size]\n",
        "test_sequences = sequences[train_size+val_size:]\n",
        "\n",
        "print(f\"Dataset split (temporal):\")\n",
        "print(f\"  Train: {len(train_sequences):,} sequences ({len(train_sequences)/n_sequences*100:.1f}%)\")\n",
        "print(f\"  Val:   {len(val_sequences):,} sequences ({len(val_sequences)/n_sequences*100:.1f}%)\")\n",
        "print(f\"  Test:  {len(test_sequences):,} sequences ({len(test_sequences)/n_sequences*100:.1f}%)\")\n",
        "print(f\"  Total: {n_sequences:,} sequences\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# PyTorch Dataset class\n",
        "class TFTDataset(Dataset):\n",
        "    def __init__(self, sequences):\n",
        "        self.sequences = sequences\n",
        "    \n",
        "    def __len__(self):\n",
        "        return len(self.sequences)\n",
        "    \n",
        "    def __getitem__(self, idx):\n",
        "        seq = self.sequences[idx]\n",
        "        \n",
        "        return {\n",
        "            'household_id': torch.LongTensor([seq['household_id']]),\n",
        "            'encoder_hour': torch.LongTensor(seq['encoder_hour']),\n",
        "            'encoder_dow': torch.LongTensor(seq['encoder_dow']),\n",
        "            'encoder_weekend': torch.FloatTensor(seq['encoder_weekend']),\n",
        "            'encoder_temp': torch.FloatTensor(seq['encoder_temp']),\n",
        "            'encoder_consumption': torch.FloatTensor(seq['encoder_consumption']),\n",
        "            'decoder_hour': torch.LongTensor(seq['decoder_hour']),\n",
        "            'decoder_dow': torch.LongTensor(seq['decoder_dow']),\n",
        "            'decoder_weekend': torch.FloatTensor(seq['decoder_weekend']),\n",
        "            'target': torch.FloatTensor(seq['target'])\n",
        "        }\n",
        "\n",
        "# Create DataLoaders\n",
        "batch_size = 64\n",
        "\n",
        "train_dataset = TFTDataset(train_sequences)\n",
        "val_dataset = TFTDataset(val_sequences)\n",
        "test_dataset = TFTDataset(test_sequences)\n",
        "\n",
        "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
        "val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)\n",
        "test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)\n",
        "\n",
        "print(f\"\\n✅ DataLoaders created (batch_size={batch_size})\")\n",
        "print(f\"  Train batches: {len(train_loader)}\")\n",
        "print(f\"  Val batches: {len(val_loader)}\")\n",
        "print(f\"  Test batches: {len(test_loader)}\")"
    ]
})

# ============================================================================
# TFT BUILDING BLOCKS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## 🏗️ Часть 3: TFT Building Blocks\n",
        "\n",
        "### 3.1 Gated Residual Network (GRN)\n",
        "\n",
        "**Key component TFT!**\n",
        "\n",
        "**Архитектура:**\n",
        "```\n",
        "Input x\n",
        "  ↓\n",
        "ELU → Linear → Dropout  ← context (optional)\n",
        "  ↓\n",
        "ELU → Linear → Dropout\n",
        "  ↓\n",
        "GLU (Gated Linear Unit)\n",
        "  ↓\n",
        "LayerNorm(x + output)  ← Residual\n",
        "```\n",
        "\n",
        "**Зачем:**\n",
        "- Gating mechanism для feature selection\n",
        "- Residual connections для gradient flow\n",
        "- Context injection (например, static features)\n"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class GatedResidualNetwork(nn.Module):\n",
        "    \"\"\"\n",
        "    Gated Residual Network (GRN)\n",
        "    \n",
        "    Key building block of TFT.\n",
        "    Applies gating mechanism with residual connections.\n",
        "    \"\"\"\n",
        "    def __init__(self, input_size, hidden_size, output_size, \n",
        "                 dropout=0.1, context_size=None):\n",
        "        super(GatedResidualNetwork, self).__init__()\n",
        "        \n",
        "        self.input_size = input_size\n",
        "        self.hidden_size = hidden_size\n",
        "        self.output_size = output_size\n",
        "        self.context_size = context_size\n",
        "        \n",
        "        # Linear layers\n",
        "        self.fc1 = nn.Linear(input_size, hidden_size)\n",
        "        \n",
        "        if context_size is not None:\n",
        "            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)\n",
        "        \n",
        "        self.fc2 = nn.Linear(hidden_size, hidden_size)\n",
        "        \n",
        "        # Gated Linear Unit\n",
        "        self.gate_fc = nn.Linear(hidden_size, output_size)\n",
        "        \n",
        "        # Skip connection\n",
        "        if input_size != output_size:\n",
        "            self.skip_fc = nn.Linear(input_size, output_size)\n",
        "        else:\n",
        "            self.skip_fc = None\n",
        "        \n",
        "        self.dropout = nn.Dropout(dropout)\n",
        "        self.layer_norm = nn.LayerNorm(output_size)\n",
        "    \n",
        "    def forward(self, x, context=None):\n",
        "        \"\"\"\n",
        "        Args:\n",
        "            x: (batch, ..., input_size)\n",
        "            context: optional (batch, ..., context_size)\n",
        "        \n",
        "        Returns:\n",
        "            output: (batch, ..., output_size)\n",
        "        \"\"\"\n",
        "        # Feed-forward\n",
        "        hidden = F.elu(self.fc1(x))\n",
        "        \n",
        "        # Context injection\n",
        "        if context is not None and self.context_size is not None:\n",
        "            hidden = hidden + self.context_fc(context)\n",
        "        \n",
        "        hidden = self.dropout(hidden)\n",
        "        hidden = F.elu(self.fc2(hidden))\n",
        "        hidden = self.dropout(hidden)\n",
        "        \n",
        "        # Gating (GLU - Gated Linear Unit)\n",
        "        gate = torch.sigmoid(self.gate_fc(hidden))\n",
        "        \n",
        "        # Skip connection\n",
        "        if self.skip_fc is not None:\n",
        "            x = self.skip_fc(x)\n",
        "        \n",
        "        # Residual + LayerNorm\n",
        "        output = self.layer_norm(x + gate * hidden)\n",
        "        \n",
        "        return output\n",
        "\n",
        "print(\"✅ GatedResidualNetwork implemented\")"
    ]
})

# ============================================================================
# VARIABLE SELECTION NETWORK
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.2 Variable Selection Network\n",
        "\n",
        "**Automatic feature selection!**\n",
        "\n",
        "**Проблема:** Не все features важны для каждого prediction\n",
        "\n",
        "**Решение:**\n",
        "1. Transform каждую feature через GRN\n",
        "2. Compute variable weights (softmax)\n",
        "3. Weighted combination\n",
        "\n",
        "**Формула:**\n",
        "$$\\text{weights} = \\text{softmax}(\\text{GRN}([\\text{flatten}(\\text{features})]))$$\n",
        "$$\\text{output} = \\sum_i \\text{weights}_i \\cdot \\text{GRN}(\\text{feature}_i)$$"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class VariableSelectionNetwork(nn.Module):\n",
        "    \"\"\"\n",
        "    Variable Selection Network\n",
        "    \n",
        "    Automatically selects relevant features.\n",
        "    \"\"\"\n",
        "    def __init__(self, input_sizes, hidden_size, output_size, dropout=0.1):\n",
        "        \"\"\"\n",
        "        Args:\n",
        "            input_sizes: list of input sizes for each variable\n",
        "            hidden_size: hidden dimension\n",
        "            output_size: output dimension\n",
        "        \"\"\"\n",
        "        super(VariableSelectionNetwork, self).__init__()\n",
        "        \n",
        "        self.num_vars = len(input_sizes)\n",
        "        self.hidden_size = hidden_size\n",
        "        self.output_size = output_size\n",
        "        \n",
        "        # GRN для каждой variable\n",
        "        self.variable_grns = nn.ModuleList([\n",
        "            GatedResidualNetwork(input_size, hidden_size, output_size, dropout)\n",
        "            for input_size in input_sizes\n",
        "        ])\n",
        "        \n",
        "        # GRN для variable selection weights\n",
        "        total_input_size = sum(input_sizes)\n",
        "        self.weight_grn = GatedResidualNetwork(\n",
        "            total_input_size, hidden_size, self.num_vars, dropout\n",
        "        )\n",
        "    \n",
        "    def forward(self, variables, return_weights=False):\n",
        "        \"\"\"\n",
        "        Args:\n",
        "            variables: list of tensors, each (batch, ..., input_size_i)\n",
        "            return_weights: whether to return variable importance weights\n",
        "        \n",
        "        Returns:\n",
        "            output: (batch, ..., output_size)\n",
        "            weights: optional (batch, num_vars)\n",
        "        \"\"\"\n",
        "        # Transform each variable через GRN\n",
        "        transformed = [grn(var) for grn, var in zip(self.variable_grns, variables)]\n",
        "        \n",
        "        # Concatenate для weight computation\n",
        "        flattened = torch.cat([var.flatten(start_dim=1) for var in variables], dim=-1)\n",
        "        \n",
        "        # Compute variable weights\n",
        "        weights = self.weight_grn(flattened)  # (batch, num_vars)\n",
        "        weights = F.softmax(weights, dim=-1)  # normalize\n",
        "        \n",
        "        # Weighted sum\n",
        "        # Stack: (batch, num_vars, ..., output_size)\n",
        "        stacked = torch.stack(transformed, dim=1)\n",
        "        \n",
        "        # Expand weights для broadcasting\n",
        "        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # (batch, num_vars, 1, 1)\n",
        "        \n",
        "        # Weighted combination\n",
        "        output = (stacked * weights_expanded).sum(dim=1)  # (batch, ..., output_size)\n",
        "        \n",
        "        if return_weights:\n",
        "            return output, weights\n",
        "        return output\n",
        "\n",
        "print(\"✅ VariableSelectionNetwork implemented\")"
    ]
})

# ============================================================================
# MULTI-HEAD ATTENTION (re-use from Step 1)
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.3 Multi-Head Attention (from Phase 4 Step 1)\n",
        "\n",
        "Переиспользуем компоненты из Step 1!"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class ScaledDotProductAttention(nn.Module):\n",
        "    \"\"\"Scaled Dot-Product Attention\"\"\"\n",
        "    def __init__(self, dropout=0.1):\n",
        "        super(ScaledDotProductAttention, self).__init__()\n",
        "        self.dropout = nn.Dropout(dropout)\n",
        "    \n",
        "    def forward(self, Q, K, V, mask=None):\n",
        "        d_k = Q.size(-1)\n",
        "        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)\n",
        "        \n",
        "        if mask is not None:\n",
        "            scores = scores.masked_fill(mask == 0, -1e9)\n",
        "        \n",
        "        attention_weights = F.softmax(scores, dim=-1)\n",
        "        attention_weights = self.dropout(attention_weights)\n",
        "        \n",
        "        context = torch.matmul(attention_weights, V)\n",
        "        return context, attention_weights\n",
        "\n",
        "\n",
        "class MultiHeadAttention(nn.Module):\n",
        "    \"\"\"Multi-Head Attention\"\"\"\n",
        "    def __init__(self, d_model, n_heads, dropout=0.1):\n",
        "        super(MultiHeadAttention, self).__init__()\n",
        "        assert d_model % n_heads == 0\n",
        "        \n",
        "        self.d_model = d_model\n",
        "        self.n_heads = n_heads\n",
        "        self.d_k = d_model // n_heads\n",
        "        \n",
        "        self.W_Q = nn.Linear(d_model, d_model)\n",
        "        self.W_K = nn.Linear(d_model, d_model)\n",
        "        self.W_V = nn.Linear(d_model, d_model)\n",
        "        \n",
        "        self.attention = ScaledDotProductAttention(dropout)\n",
        "        self.W_O = nn.Linear(d_model, d_model)\n",
        "        self.dropout = nn.Dropout(dropout)\n",
        "    \n",
        "    def split_heads(self, x):\n",
        "        batch_size, seq_len, d_model = x.size()\n",
        "        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)\n",
        "    \n",
        "    def combine_heads(self, x):\n",
        "        batch_size, n_heads, seq_len, d_k = x.size()\n",
        "        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)\n",
        "    \n",
        "    def forward(self, Q, K, V, mask=None):\n",
        "        Q = self.split_heads(self.W_Q(Q))\n",
        "        K = self.split_heads(self.W_K(K))\n",
        "        V = self.split_heads(self.W_V(V))\n",
        "        \n",
        "        context, attention_weights = self.attention(Q, K, V, mask)\n",
        "        context = self.combine_heads(context)\n",
        "        output = self.W_O(context)\n",
        "        output = self.dropout(output)\n",
        "        \n",
        "        return output, attention_weights\n",
        "\n",
        "print(\"✅ Multi-Head Attention loaded (from Phase 4 Step 1)\")"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.4 Test Building Blocks"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Test GRN\n",
        "print(\"Testing GatedResidualNetwork...\")\n",
        "grn = GatedResidualNetwork(input_size=10, hidden_size=32, output_size=16)\n",
        "x_test = torch.randn(4, 5, 10)  # (batch=4, seq=5, features=10)\n",
        "out = grn(x_test)\n",
        "print(f\"  Input: {x_test.shape} → Output: {out.shape}\")\n",
        "assert out.shape == (4, 5, 16), \"GRN output shape mismatch\"\n",
        "print(\"  ✅ GRN works!\")\n",
        "\n",
        "# Test VSN\n",
        "print(\"\\nTesting VariableSelectionNetwork...\")\n",
        "vsn = VariableSelectionNetwork(input_sizes=[10, 15, 20], hidden_size=32, output_size=16)\n",
        "vars_test = [\n",
        "    torch.randn(4, 5, 10),\n",
        "    torch.randn(4, 5, 15),\n",
        "    torch.randn(4, 5, 20)\n",
        "]\n",
        "out, weights = vsn(vars_test, return_weights=True)\n",
        "print(f\"  Output: {out.shape}\")\n",
        "print(f\"  Weights: {weights.shape}\")\n",
        "print(f\"  Weights sum: {weights[0].sum().item():.4f} (should be 1.0)\")\n",
        "assert out.shape == (4, 5, 16), \"VSN output shape mismatch\"\n",
        "assert weights.shape == (4, 3), \"VSN weights shape mismatch\"\n",
        "print(\"  ✅ VSN works!\")\n",
        "\n",
        "# Test MHA\n",
        "print(\"\\nTesting MultiHeadAttention...\")\n",
        "mha = MultiHeadAttention(d_model=64, n_heads=4)\n",
        "x_test = torch.randn(4, 10, 64)  # (batch=4, seq=10, d_model=64)\n",
        "out, attn = mha(x_test, x_test, x_test)\n",
        "print(f\"  Input: {x_test.shape} → Output: {out.shape}\")\n",
        "print(f\"  Attention weights: {attn.shape}\")\n",
        "assert out.shape == (4, 10, 64), \"MHA output shape mismatch\"\n",
        "print(\"  ✅ MHA works!\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"✅ All TFT building blocks implemented and tested!\")\n",
        "print(\"=\"*60)"
    ]
})

# Сохраняем
notebook['cells'] = cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Part 2 добавлена в: {notebook_path}')
print(f'Всего ячеек: {len(cells)}')
print('Следующая часть: Full TFT Model, Training, Evaluation...')
