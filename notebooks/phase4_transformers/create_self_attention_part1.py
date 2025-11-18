#!/usr/bin/env python3
"""
Phase 4 Step 1: Self-Attention & Transformer Basics
Part 1: Introduction, Theory, Scaled Dot-Product Attention
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
        "# 🔮 Self-Attention & Transformer Basics\n",
        "\n",
        "**Phase 4, Step 1: Transformers & Modern Architectures**\n",
        "\n",
        "---\n",
        "\n",
        "## 🎯 Цель этого ноутбука\n",
        "\n",
        "В Phase 3 мы изучили **RNN/LSTM + Attention**:\n",
        "- ✅ Recurrent архитектуры для последовательностей\n",
        "- ✅ Attention как механизм взвешивания важности\n",
        "- ✅ Seq2Seq модели\n",
        "\n",
        "**Но у RNN есть проблемы:**\n",
        "- ❌ **Sequential processing**: нельзя параллелить\n",
        "- ❌ **Vanishing gradients**: сложно учить длинные зависимости\n",
        "- ❌ **Slow training**: обработка по одному timestep\n",
        "\n",
        "---\n",
        "\n",
        "## 🚀 Enter Transformers (2017)\n",
        "\n",
        "**\"Attention is All You Need\"** (Vaswani et al., 2017)\n",
        "\n",
        "**Ключевая идея:** Полностью избавиться от рекуррентности!\n",
        "- ✅ **Self-Attention**: каждый элемент смотрит на все остальные одновременно\n",
        "- ✅ **Parallelization**: все вычисления параллельны\n",
        "- ✅ **Long-range dependencies**: прямые связи между любыми элементами\n",
        "- ✅ **Scalability**: эффективно на GPU/TPU\n",
        "\n",
        "**Результат:**\n",
        "- 🏆 SOTA в NLP: BERT, GPT, T5, GPT-3/4\n",
        "- 🏆 Computer Vision: ViT (Vision Transformer), DINO\n",
        "- 🏆 Tabular Data: TabTransformer, FT-Transformer\n",
        "- 🏆 Time Series: Temporal Fusion Transformer\n",
        "- 🏆 Multi-modal: CLIP, Flamingo\n",
        "\n",
        "---\n",
        "\n",
        "## 📚 Что мы изучим\n",
        "\n",
        "### 1. Self-Attention Mechanism\n",
        "- **Query, Key, Value (Q, K, V)**: как работает attention\n",
        "- **Scaled Dot-Product Attention**: математика\n",
        "- **Attention Weights**: что модель \"смотрит\"\n",
        "- **Implementation**: с нуля в PyTorch\n",
        "\n",
        "### 2. Multi-Head Attention\n",
        "- **Multiple attention heads**: параллельные \"perspectives\"\n",
        "- **Concatenation & projection**: объединение heads\n",
        "- **Why it works**: разные heads учат разные паттерны\n",
        "\n",
        "### 3. Positional Encoding\n",
        "- **Problem**: Self-Attention permutation-invariant\n",
        "- **Solution**: добавить информацию о позиции\n",
        "- **Sinusoidal encoding**: для sequences\n",
        "- **Learnable embeddings**: для табличных данных\n",
        "\n",
        "### 4. Transformer Encoder for Tabular Data\n",
        "- **Dataset**: Titanic (классификация выживших)\n",
        "- **Architecture**: Feature Embedding → Multi-Head Attention → FFN\n",
        "- **Training**: Cross-Entropy Loss\n",
        "- **Evaluation**: сравнение с XGBoost, LSTM\n",
        "- **Interpretability**: визуализация attention weights\n",
        "\n",
        "---\n",
        "\n",
        "## 🔍 Почему Transformers для табличных данных?\n",
        "\n",
        "**Традиционный подход:**\n",
        "- Tree-based (XGBoost, LightGBM): хорошо для табличных данных\n",
        "- MLPs: baseline\n",
        "\n",
        "**Transformers дают:**\n",
        "- ✅ **Feature interactions**: автоматическое изучение взаимодействий\n",
        "- ✅ **Attention weights**: интерпретируемость\n",
        "- ✅ **Transfer learning**: pre-training на больших датасетах\n",
        "- ✅ **Mixed data types**: категориальные + числовые\n",
        "\n",
        "**Когда использовать:**\n",
        "- 📊 Большие датасеты (>10k samples)\n",
        "- 📊 Много categorical features\n",
        "- 📊 Сложные feature interactions\n",
        "- 📊 Нужна интерпретируемость\n",
        "\n",
        "---\n"
    ]
})

# ============================================================================
# IMPORTS
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 💻 Часть 1: Теория и имплементация Self-Attention\n",
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
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# PyTorch\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import torch.nn.functional as F\n",
        "from torch.utils.data import Dataset, DataLoader, TensorDataset\n",
        "\n",
        "# Sklearn\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
        "from sklearn.metrics import (\n",
        "    accuracy_score, precision_score, recall_score, f1_score,\n",
        "    classification_report, confusion_matrix, roc_auc_score\n",
        ")\n",
        "\n",
        "# Math\n",
        "import math\n",
        "\n",
        "# Настройки\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "sns.set_palette(\"husl\")\n",
        "%matplotlib inline\n",
        "\n",
        "# Device\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f\"Device: {device}\")\n",
        "\n",
        "# Reproducibility\n",
        "torch.manual_seed(42)\n",
        "np.random.seed(42)\n",
        "\n",
        "print(\"\\n✅ Все библиотеки загружены\")"
    ]
})

# ============================================================================
# ATTENTION THEORY
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.2 Теория: Что такое Self-Attention?\n",
        "\n",
        "---\n",
        "\n",
        "## 🧠 Интуиция\n",
        "\n",
        "**Представьте предложение:** \"The animal didn't cross the street because **it** was too tired.\"\n",
        "\n",
        "**Вопрос:** На что ссылается \"it\"?\n",
        "- Ответ: \"The animal\" (а не \"street\")\n",
        "\n",
        "**Self-Attention делает именно это:**\n",
        "- Для каждого слова смотрит на **все остальные слова**\n",
        "- Вычисляет **веса важности** (насколько важно каждое слово)\n",
        "- Создает **контекстное представление** взвешенной суммой\n",
        "\n",
        "---\n",
        "\n",
        "## 📐 Математика: Scaled Dot-Product Attention\n",
        "\n",
        "**Input:**\n",
        "- Sequence: $X = [x_1, x_2, ..., x_n]$, где $x_i \\in \\mathbb{R}^{d}$\n",
        "\n",
        "**Шаг 1: Создаем Q, K, V (Query, Key, Value)**\n",
        "\n",
        "$$Q = XW^Q, \\quad K = XW^K, \\quad V = XW^V$$\n",
        "\n",
        "Где:\n",
        "- $W^Q, W^K, W^V \\in \\mathbb{R}^{d \\times d_k}$ - learnable matrices\n",
        "- $Q, K, V \\in \\mathbb{R}^{n \\times d_k}$\n",
        "\n",
        "**Интуиция:**\n",
        "- **Query (Q)**: \"Что я ищу?\" (запрос от текущего элемента)\n",
        "- **Key (K)**: \"Что я могу предложить?\" (описание других элементов)\n",
        "- **Value (V)**: \"Какую информацию я несу?\" (actual content)\n",
        "\n",
        "**Шаг 2: Вычисляем Attention Scores**\n",
        "\n",
        "$$\\text{scores} = \\frac{QK^T}{\\sqrt{d_k}}$$\n",
        "\n",
        "- $QK^T$: similarity между queries и keys (dot product)\n",
        "- $\\sqrt{d_k}$: scaling для стабильности градиентов\n",
        "\n",
        "**Почему scaling?**\n",
        "- Без scaling: для больших $d_k$, dot products огромные\n",
        "- Огромные scores → softmax saturation → vanishing gradients\n",
        "- $\\sqrt{d_k}$ нормализует variance\n",
        "\n",
        "**Шаг 3: Softmax для весов**\n",
        "\n",
        "$$\\text{weights} = \\text{softmax}(\\text{scores}) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)$$\n",
        "\n",
        "- Преобразует scores в вероятности: $\\sum_i w_i = 1$\n",
        "- Высокие scores → высокие веса\n",
        "\n",
        "**Шаг 4: Weighted Sum**\n",
        "\n",
        "$$\\text{Attention}(Q, K, V) = \\text{weights} \\cdot V = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V$$\n",
        "\n",
        "- Каждый output - взвешенная комбинация всех values\n",
        "- Веса определяют, сколько каждый элемент \"смотрит\" на другие\n",
        "\n",
        "---\n",
        "\n",
        "## 🎨 Визуальная интуиция\n",
        "\n",
        "```\n",
        "Input:     [x1]  [x2]  [x3]  [x4]\n",
        "              ↓     ↓     ↓     ↓\n",
        "           [Q1]  [Q2]  [Q3]  [Q4]  ← Queries (\"что я ищу?\")\n",
        "           [K1]  [K2]  [K3]  [K4]  ← Keys (\"что я предлагаю?\")\n",
        "           [V1]  [V2]  [V3]  [V4]  ← Values (actual info)\n",
        "\n",
        "Attention for x1:\n",
        "  Q1 · K1 → score11  ┐\n",
        "  Q1 · K2 → score12  ├→ softmax → [w11, w12, w13, w14]\n",
        "  Q1 · K3 → score13  │\n",
        "  Q1 · K4 → score14  ┘\n",
        "\n",
        "Output: y1 = w11*V1 + w12*V2 + w13*V3 + w14*V4\n",
        "```\n",
        "\n",
        "---\n",
        "\n",
        "## 🔑 Ключевые свойства\n",
        "\n",
        "1. **Permutation Invariant (без Positional Encoding):**\n",
        "   - Attention не зависит от порядка входов\n",
        "   - $[x_1, x_2, x_3] \\equiv [x_3, x_1, x_2]$\n",
        "   - Нужно добавлять positional encoding!\n",
        "\n",
        "2. **Parallelizable:**\n",
        "   - Все attention scores вычисляются одновременно\n",
        "   - Матричное умножение $QK^T$ - одна операция\n",
        "   - Нет sequential dependencies (в отличие от RNN)\n",
        "\n",
        "3. **Long-range Dependencies:**\n",
        "   - Прямая связь между любыми элементами\n",
        "   - O(1) path length (vs O(n) в RNN)\n",
        "\n",
        "4. **Computational Complexity:**\n",
        "   - $O(n^2 \\cdot d)$ для sequence length $n$\n",
        "   - Bottleneck для очень длинных последовательностей\n",
        "   - Решение: Sparse Attention, Linformer, etc.\n",
        "\n",
        "---\n"
    ]
})

# ============================================================================
# IMPLEMENTATION: Scaled Dot-Product Attention
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.3 Implementation: Scaled Dot-Product Attention\n",
        "\n",
        "Имплементируем с нуля!"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class ScaledDotProductAttention(nn.Module):\n",
        "    \"\"\"\n",
        "    Scaled Dot-Product Attention\n",
        "    \n",
        "    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V\n",
        "    \"\"\"\n",
        "    def __init__(self, dropout=0.1):\n",
        "        super(ScaledDotProductAttention, self).__init__()\n",
        "        self.dropout = nn.Dropout(dropout)\n",
        "    \n",
        "    def forward(self, Q, K, V, mask=None):\n",
        "        \"\"\"\n",
        "        Args:\n",
        "            Q: Query matrix (batch_size, n_heads, seq_len, d_k)\n",
        "            K: Key matrix (batch_size, n_heads, seq_len, d_k)\n",
        "            V: Value matrix (batch_size, n_heads, seq_len, d_v)\n",
        "            mask: Mask matrix (optional)\n",
        "        \n",
        "        Returns:\n",
        "            context: Attention output (batch_size, n_heads, seq_len, d_v)\n",
        "            attention_weights: Attention weights (batch_size, n_heads, seq_len, seq_len)\n",
        "        \"\"\"\n",
        "        # d_k: dimension of keys/queries\n",
        "        d_k = Q.size(-1)\n",
        "        \n",
        "        # Шаг 1: Compute attention scores\n",
        "        # scores shape: (batch_size, n_heads, seq_len, seq_len)\n",
        "        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)\n",
        "        \n",
        "        # Шаг 2: Apply mask (если есть)\n",
        "        if mask is not None:\n",
        "            scores = scores.masked_fill(mask == 0, -1e9)\n",
        "        \n",
        "        # Шаг 3: Apply softmax\n",
        "        attention_weights = F.softmax(scores, dim=-1)\n",
        "        attention_weights = self.dropout(attention_weights)\n",
        "        \n",
        "        # Шаг 4: Weighted sum of values\n",
        "        context = torch.matmul(attention_weights, V)\n",
        "        \n",
        "        return context, attention_weights\n",
        "\n",
        "print(\"✅ ScaledDotProductAttention реализован!\")"
    ]
})

# ============================================================================
# SIMPLE EXAMPLE
# ============================================================================

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 1.4 Пример: Attention на простых данных\n",
        "\n",
        "Создадим маленький пример, чтобы понять, как работает attention."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Создаем простой пример\n",
        "batch_size = 1\n",
        "n_heads = 1  # пока один head\n",
        "seq_len = 4  # 4 элемента в последовательности\n",
        "d_k = 8      # размерность keys/queries\n",
        "\n",
        "# Случайные Q, K, V\n",
        "torch.manual_seed(42)\n",
        "Q = torch.randn(batch_size, n_heads, seq_len, d_k)\n",
        "K = torch.randn(batch_size, n_heads, seq_len, d_k)\n",
        "V = torch.randn(batch_size, n_heads, seq_len, d_k)\n",
        "\n",
        "print(f\"Q shape: {Q.shape}\")\n",
        "print(f\"K shape: {K.shape}\")\n",
        "print(f\"V shape: {V.shape}\")\n",
        "\n",
        "# Применяем attention\n",
        "attention_layer = ScaledDotProductAttention(dropout=0.0)\n",
        "context, attention_weights = attention_layer(Q, K, V)\n",
        "\n",
        "print(f\"\\nContext shape: {context.shape}\")\n",
        "print(f\"Attention weights shape: {attention_weights.shape}\")\n",
        "\n",
        "# Визуализируем attention weights\n",
        "weights = attention_weights[0, 0].detach().numpy()  # (seq_len, seq_len)\n",
        "\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.heatmap(weights, annot=True, fmt='.3f', cmap='YlOrRd', \n",
        "            xticklabels=[f'K{i+1}' for i in range(seq_len)],\n",
        "            yticklabels=[f'Q{i+1}' for i in range(seq_len)],\n",
        "            cbar_kws={'label': 'Attention Weight'})\n",
        "plt.title('Attention Weights Matrix', fontsize=16, fontweight='bold')\n",
        "plt.xlabel('Keys (what to attend to)', fontsize=12)\n",
        "plt.ylabel('Queries (who is attending)', fontsize=12)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n📊 Интерпретация:\")\n",
        "print(\"  - Каждая строка: как Q_i распределяет внимание на все Keys\")\n",
        "print(\"  - Каждая строка суммируется в 1.0 (softmax property)\")\n",
        "print(\"  - Высокие веса: Q_i сильно \\\"смотрит\\\" на K_j\")\n",
        "\n",
        "# Проверяем, что строки суммируются в 1\n",
        "row_sums = weights.sum(axis=1)\n",
        "print(f\"\\n✅ Проверка softmax: Row sums = {row_sums}\")"
    ]
})

# Сохраняем первую часть
notebook['cells'] = cells

output_path = '/home/user/test/notebooks/phase4_transformers/01_self_attention_transformer.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Part 1 создана: {output_path}')
print(f'Ячеек: {len(cells)}')
print('Следующая часть: Multi-Head Attention и Positional Encoding...')
