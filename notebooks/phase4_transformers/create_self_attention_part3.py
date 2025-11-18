#!/usr/bin/env python3
"""
Phase 4 Step 1: Self-Attention & Transformer Basics
Part 3: Transformer Encoder and Titanic Dataset Preparation
"""

import json

# Загружаем существующий notebook
notebook_path = '/home/user/test/notebooks/phase4_transformers/01_self_attention_transformer.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

cells = notebook['cells']

# ============================================================================
# TRANSFORMER ENCODER THEORY
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## 🏗️ Часть 4: Transformer Encoder\n",
        "\n",
        "### 4.1 Теория: Полная архитектура Encoder\n",
        "\n",
        "---\n",
        "\n",
        "## 📐 Transformer Encoder Block\n",
        "\n",
        "**Структура одного Encoder Block:**\n",
        "\n",
        "```\n",
        "Input\n",
        "  ↓\n",
        "┌─────────────────────┐\n",
        "│ Multi-Head Attention│\n",
        "└──────────┬──────────┘\n",
        "           ↓\n",
        "      Add & Norm (Residual + LayerNorm)\n",
        "           ↓\n",
        "┌─────────────────────┐\n",
        "│  Feed Forward (FFN) │\n",
        "│   (2-layer MLP)     │\n",
        "└──────────┬──────────┘\n",
        "           ↓\n",
        "      Add & Norm (Residual + LayerNorm)\n",
        "           ↓\n",
        "        Output\n",
        "```\n",
        "\n",
        "**Компоненты:**\n",
        "\n",
        "1. **Multi-Head Self-Attention:**\n",
        "   - Q = K = V = Input (self-attention)\n",
        "   - Feature interactions\n",
        "\n",
        "2. **Add & Norm (Residual Connection + Layer Normalization):**\n",
        "   $$\\text{LayerNorm}(x + \\text{Sublayer}(x))$$\n",
        "   - Residual: помогает gradient flow\n",
        "   - LayerNorm: стабилизирует обучение\n",
        "\n",
        "3. **Feed Forward Network (FFN):**\n",
        "   $$\\text{FFN}(x) = \\text{ReLU}(xW_1 + b_1)W_2 + b_2$$\n",
        "   - 2-layer MLP с ReLU\n",
        "   - Применяется independently к каждой позиции\n",
        "   - Обычно: hidden dim = 4 × d_model\n",
        "\n",
        "4. **Another Add & Norm**\n",
        "\n",
        "**Stacking Multiple Blocks:**\n",
        "- Original Transformer: 6 encoder blocks\n",
        "- BERT-base: 12 blocks\n",
        "- GPT-3: 96 blocks\n",
        "\n",
        "---\n",
        "\n",
        "## 🎯 Для табличных данных\n",
        "\n",
        "**Архитектура:**\n",
        "\n",
        "```\n",
        "Tabular Features [x1, x2, ..., xn]\n",
        "        ↓\n",
        "Feature Embedding (Linear projection)\n",
        "        ↓\n",
        "Positional Encoding (learnable)\n",
        "        ↓\n",
        "Transformer Encoder Blocks (N layers)\n",
        "        ↓\n",
        "Global Pooling (mean/max/CLS token)\n",
        "        ↓\n",
        "Classification Head (Linear)\n",
        "        ↓\n",
        "Output (class probabilities)\n",
        "```\n",
        "\n",
        "**Key Differences from NLP:**\n",
        "- **No word embeddings**: Linear projection вместо embedding lookup\n",
        "- **Learnable PE**: вместо sinusoidal (нет естественного порядка)\n",
        "- **Global pooling**: mean/max вместо CLS token (можно и CLS)\n",
        "\n",
        "---\n"
    ]
})

# ============================================================================
# FEED FORWARD NETWORK
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 4.2 Implementation: Feed Forward Network"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class FeedForward(nn.Module):\n",
        "    \"\"\"\n",
        "    Position-wise Feed Forward Network\n",
        "    \n",
        "    FFN(x) = ReLU(xW1 + b1)W2 + b2\n",
        "    \"\"\"\n",
        "    def __init__(self, d_model, d_ff, dropout=0.1):\n",
        "        \"\"\"\n",
        "        Args:\n",
        "            d_model: Model dimension\n",
        "            d_ff: Hidden dimension (usually 4 * d_model)\n",
        "            dropout: Dropout rate\n",
        "        \"\"\"\n",
        "        super(FeedForward, self).__init__()\n",
        "        self.linear1 = nn.Linear(d_model, d_ff)\n",
        "        self.linear2 = nn.Linear(d_ff, d_model)\n",
        "        self.dropout = nn.Dropout(dropout)\n",
        "    \n",
        "    def forward(self, x):\n",
        "        \"\"\"\n",
        "        Args:\n",
        "            x: (batch_size, seq_len, d_model)\n",
        "        \n",
        "        Returns:\n",
        "            (batch_size, seq_len, d_model)\n",
        "        \"\"\"\n",
        "        return self.linear2(self.dropout(F.relu(self.linear1(x))))\n",
        "\n",
        "print(\"✅ FeedForward реализован!\")"
    ]
})

# ============================================================================
# TRANSFORMER ENCODER BLOCK
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 4.3 Implementation: Transformer Encoder Block"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class TransformerEncoderBlock(nn.Module):\n",
        "    \"\"\"\n",
        "    Single Transformer Encoder Block\n",
        "    \n",
        "    Components:\n",
        "    1. Multi-Head Self-Attention\n",
        "    2. Add & Norm\n",
        "    3. Feed Forward Network\n",
        "    4. Add & Norm\n",
        "    \"\"\"\n",
        "    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):\n",
        "        \"\"\"\n",
        "        Args:\n",
        "            d_model: Model dimension\n",
        "            n_heads: Number of attention heads\n",
        "            d_ff: Feed-forward hidden dimension\n",
        "            dropout: Dropout rate\n",
        "        \"\"\"\n",
        "        super(TransformerEncoderBlock, self).__init__()\n",
        "        \n",
        "        # Multi-Head Attention\n",
        "        self.attention = MultiHeadAttention(d_model, n_heads, dropout)\n",
        "        \n",
        "        # Feed Forward\n",
        "        self.feed_forward = FeedForward(d_model, d_ff, dropout)\n",
        "        \n",
        "        # Layer Normalization\n",
        "        self.norm1 = nn.LayerNorm(d_model)\n",
        "        self.norm2 = nn.LayerNorm(d_model)\n",
        "        \n",
        "        # Dropout\n",
        "        self.dropout = nn.Dropout(dropout)\n",
        "    \n",
        "    def forward(self, x, mask=None):\n",
        "        \"\"\"\n",
        "        Args:\n",
        "            x: (batch_size, seq_len, d_model)\n",
        "            mask: Optional mask\n",
        "        \n",
        "        Returns:\n",
        "            (batch_size, seq_len, d_model)\n",
        "        \"\"\"\n",
        "        # 1. Multi-Head Attention + Add & Norm\n",
        "        attn_output, attention_weights = self.attention(x, x, x, mask)\n",
        "        x = self.norm1(x + self.dropout(attn_output))\n",
        "        \n",
        "        # 2. Feed Forward + Add & Norm\n",
        "        ff_output = self.feed_forward(x)\n",
        "        x = self.norm2(x + self.dropout(ff_output))\n",
        "        \n",
        "        return x, attention_weights\n",
        "\n",
        "print(\"✅ TransformerEncoderBlock реализован!\")"
    ]
})

# ============================================================================
# FULL TRANSFORMER ENCODER
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 4.4 Implementation: Full Transformer Encoder для табличных данных"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class TabularTransformerEncoder(nn.Module):\n",
        "    \"\"\"\n",
        "    Transformer Encoder для табличных данных\n",
        "    \n",
        "    Architecture:\n",
        "    Input → Feature Embedding → Positional Encoding → \n",
        "    Transformer Blocks → Global Pooling → Classification Head\n",
        "    \"\"\"\n",
        "    def __init__(self, num_features, d_model, n_heads, n_layers, d_ff, \n",
        "                 num_classes, dropout=0.1, pooling='mean'):\n",
        "        \"\"\"\n",
        "        Args:\n",
        "            num_features: Number of input features\n",
        "            d_model: Model dimension\n",
        "            n_heads: Number of attention heads\n",
        "            n_layers: Number of Transformer blocks\n",
        "            d_ff: Feed-forward hidden dimension\n",
        "            num_classes: Number of output classes\n",
        "            dropout: Dropout rate\n",
        "            pooling: Pooling method ('mean', 'max', 'cls')\n",
        "        \"\"\"\n",
        "        super(TabularTransformerEncoder, self).__init__()\n",
        "        \n",
        "        self.num_features = num_features\n",
        "        self.d_model = d_model\n",
        "        self.pooling = pooling\n",
        "        \n",
        "        # Feature embedding (project each feature to d_model)\n",
        "        self.feature_embedding = nn.Linear(1, d_model)  # each feature independently\n",
        "        \n",
        "        # Positional encoding (learnable)\n",
        "        self.pos_encoding = LearnablePositionalEmbedding(num_features, d_model, dropout)\n",
        "        \n",
        "        # Stack of Transformer Encoder blocks\n",
        "        self.encoder_blocks = nn.ModuleList([\n",
        "            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)\n",
        "            for _ in range(n_layers)\n",
        "        ])\n",
        "        \n",
        "        # CLS token (if using cls pooling)\n",
        "        if pooling == 'cls':\n",
        "            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))\n",
        "        \n",
        "        # Classification head\n",
        "        self.classifier = nn.Sequential(\n",
        "            nn.Linear(d_model, d_model // 2),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(dropout),\n",
        "            nn.Linear(d_model // 2, num_classes)\n",
        "        )\n",
        "    \n",
        "    def forward(self, x, return_attention=False):\n",
        "        \"\"\"\n",
        "        Args:\n",
        "            x: (batch_size, num_features)\n",
        "            return_attention: Whether to return attention weights\n",
        "        \n",
        "        Returns:\n",
        "            logits: (batch_size, num_classes)\n",
        "            attention_weights: Optional, list of attention weights from each layer\n",
        "        \"\"\"\n",
        "        batch_size = x.size(0)\n",
        "        \n",
        "        # 1. Feature embedding: (batch, num_features) → (batch, num_features, d_model)\n",
        "        x = x.unsqueeze(-1)  # (batch, num_features, 1)\n",
        "        x = self.feature_embedding(x)  # (batch, num_features, d_model)\n",
        "        \n",
        "        # 2. Add CLS token if using cls pooling\n",
        "        if self.pooling == 'cls':\n",
        "            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)\n",
        "            x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_features+1, d_model)\n",
        "        \n",
        "        # 3. Add positional encoding\n",
        "        x = self.pos_encoding(x)\n",
        "        \n",
        "        # 4. Pass through Transformer blocks\n",
        "        attention_weights_list = []\n",
        "        for encoder_block in self.encoder_blocks:\n",
        "            x, attn_weights = encoder_block(x)\n",
        "            if return_attention:\n",
        "                attention_weights_list.append(attn_weights)\n",
        "        \n",
        "        # 5. Global pooling\n",
        "        if self.pooling == 'mean':\n",
        "            x = x.mean(dim=1)  # (batch, d_model)\n",
        "        elif self.pooling == 'max':\n",
        "            x = x.max(dim=1)[0]  # (batch, d_model)\n",
        "        elif self.pooling == 'cls':\n",
        "            x = x[:, 0, :]  # take CLS token\n",
        "        \n",
        "        # 6. Classification\n",
        "        logits = self.classifier(x)\n",
        "        \n",
        "        if return_attention:\n",
        "            return logits, attention_weights_list\n",
        "        return logits\n",
        "\n",
        "print(\"✅ TabularTransformerEncoder реализован!\")\n",
        "print(\"\\nАрхитектура готова:\")\n",
        "print(\"  1. Feature Embedding (Linear projection)\")\n",
        "print(\"  2. Learnable Positional Encoding\")\n",
        "print(\"  3. Stack of Transformer Encoder Blocks\")\n",
        "print(\"  4. Global Pooling (mean/max/cls)\")\n",
        "print(\"  5. Classification Head\")"
    ]
})

# ============================================================================
# TITANIC DATA LOADING
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## 📊 Часть 5: Titanic Dataset\n",
        "\n",
        "### 5.1 Загрузка и подготовка данных\n",
        "\n",
        "**Задача:** Предсказать выживание пассажиров Титаника  \n",
        "**Тип:** Бинарная классификация (Survived: 0 или 1)  \n",
        "**Features:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked\n"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Загрузка Titanic dataset\n",
        "# Используем встроенный датасет из seaborn\n",
        "\n",
        "df = sns.load_dataset('titanic')\n",
        "\n",
        "print(f\"Dataset shape: {df.shape}\")\n",
        "print(f\"\\nFirst few rows:\")\n",
        "print(df.head())\n",
        "\n",
        "print(f\"\\nColumns: {df.columns.tolist()}\")\n",
        "print(f\"\\nMissing values:\")\n",
        "print(df.isnull().sum())"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Feature Engineering\n",
        "\n",
        "# Выбираем нужные features\n",
        "features_to_use = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone']\n",
        "target = 'survived'\n",
        "\n",
        "# Создаем копию\n",
        "data = df[features_to_use + [target]].copy()\n",
        "\n",
        "# Handle missing values\n",
        "data['age'].fillna(data['age'].median(), inplace=True)\n",
        "data['fare'].fillna(data['fare'].median(), inplace=True)\n",
        "data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)\n",
        "\n",
        "# Encode categorical variables\n",
        "# Sex: male=1, female=0\n",
        "data['sex'] = (data['sex'] == 'male').astype(int)\n",
        "\n",
        "# Embarked: one-hot encoding\n",
        "data = pd.get_dummies(data, columns=['embarked'], prefix='embarked', drop_first=True)\n",
        "\n",
        "# alone: boolean to int\n",
        "data['alone'] = data['alone'].astype(int)\n",
        "\n",
        "print(f\"After preprocessing: {data.shape}\")\n",
        "print(f\"\\nFeatures: {[col for col in data.columns if col != 'survived']}\")\n",
        "print(f\"\\nMissing values: {data.isnull().sum().sum()}\")\n",
        "\n",
        "# Check class balance\n",
        "print(f\"\\nClass distribution:\")\n",
        "print(data['survived'].value_counts())\n",
        "print(f\"\\nSurvival rate: {data['survived'].mean():.2%}\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualize data\n",
        "fig, axes = plt.subplots(2, 3, figsize=(16, 10))\n",
        "\n",
        "# Survival by class\n",
        "pd.crosstab(data['pclass'], data['survived']).plot(kind='bar', ax=axes[0, 0])\n",
        "axes[0, 0].set_title('Survival by Class', fontweight='bold')\n",
        "axes[0, 0].set_xlabel('Pclass')\n",
        "axes[0, 0].set_ylabel('Count')\n",
        "axes[0, 0].legend(['Died', 'Survived'])\n",
        "\n",
        "# Survival by sex\n",
        "pd.crosstab(data['sex'], data['survived']).plot(kind='bar', ax=axes[0, 1])\n",
        "axes[0, 1].set_title('Survival by Sex', fontweight='bold')\n",
        "axes[0, 1].set_xlabel('Sex (0=female, 1=male)')\n",
        "axes[0, 1].set_xticklabels(['Female', 'Male'], rotation=0)\n",
        "axes[0, 1].legend(['Died', 'Survived'])\n",
        "\n",
        "# Age distribution\n",
        "data[data['survived'] == 0]['age'].hist(bins=30, alpha=0.5, label='Died', ax=axes[0, 2])\n",
        "data[data['survived'] == 1]['age'].hist(bins=30, alpha=0.5, label='Survived', ax=axes[0, 2])\n",
        "axes[0, 2].set_title('Age Distribution', fontweight='bold')\n",
        "axes[0, 2].set_xlabel('Age')\n",
        "axes[0, 2].legend()\n",
        "\n",
        "# Fare distribution\n",
        "data[data['survived'] == 0]['fare'].hist(bins=30, alpha=0.5, label='Died', ax=axes[1, 0])\n",
        "data[data['survived'] == 1]['fare'].hist(bins=30, alpha=0.5, label='Survived', ax=axes[1, 0])\n",
        "axes[1, 0].set_title('Fare Distribution', fontweight='bold')\n",
        "axes[1, 0].set_xlabel('Fare')\n",
        "axes[1, 0].legend()\n",
        "\n",
        "# SibSp distribution\n",
        "pd.crosstab(data['sibsp'], data['survived']).plot(kind='bar', ax=axes[1, 1])\n",
        "axes[1, 1].set_title('Survival by SibSp', fontweight='bold')\n",
        "axes[1, 1].set_xlabel('Siblings/Spouses')\n",
        "axes[1, 1].legend(['Died', 'Survived'])\n",
        "\n",
        "# Correlation heatmap\n",
        "corr = data.corr()\n",
        "sns.heatmap(corr[['survived']].sort_values('survived', ascending=False), \n",
        "            annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1, 2], cbar=False)\n",
        "axes[1, 2].set_title('Correlation with Survival', fontweight='bold')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n📊 Key insights:\")\n",
        "print(\"  - Women had much higher survival rate\")\n",
        "print(\"  - First class passengers survived more\")\n",
        "print(\"  - Children had better survival chances\")\n",
        "print(\"  - Higher fare → higher survival (proxy for class)\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Train/Test Split\n",
        "\n",
        "# Separate features and target\n",
        "X = data.drop('survived', axis=1).values\n",
        "y = data['survived'].values\n",
        "\n",
        "print(f\"X shape: {X.shape}\")\n",
        "print(f\"y shape: {y.shape}\")\n",
        "\n",
        "# Split\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=42, stratify=y\n",
        ")\n",
        "\n",
        "print(f\"\\nTrain set: {X_train.shape[0]} samples\")\n",
        "print(f\"Test set: {X_test.shape[0]} samples\")\n",
        "\n",
        "# Standardize features\n",
        "scaler = StandardScaler()\n",
        "X_train = scaler.fit_transform(X_train)\n",
        "X_test = scaler.transform(X_test)\n",
        "\n",
        "print(\"\\n✅ Features standardized\")\n",
        "\n",
        "# Convert to PyTorch tensors\n",
        "X_train_tensor = torch.FloatTensor(X_train)\n",
        "y_train_tensor = torch.LongTensor(y_train)\n",
        "X_test_tensor = torch.FloatTensor(X_test)\n",
        "y_test_tensor = torch.LongTensor(y_test)\n",
        "\n",
        "# Create DataLoaders\n",
        "train_dataset = TensorDataset(X_train_tensor, y_train_tensor)\n",
        "test_dataset = TensorDataset(X_test_tensor, y_test_tensor)\n",
        "\n",
        "batch_size = 32\n",
        "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
        "test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)\n",
        "\n",
        "print(f\"\\n✅ DataLoaders created (batch_size={batch_size})\")\n",
        "print(f\"Number of features: {X_train.shape[1]}\")"
    ]
})

# Сохраняем
notebook['cells'] = cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Part 3 добавлена в: {notebook_path}')
print(f'Всего ячеек: {len(cells)}')
print('Следующая часть: Training, Evaluation, Attention Visualization...')
