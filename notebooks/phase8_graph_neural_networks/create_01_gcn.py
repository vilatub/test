#!/usr/bin/env python3
"""
Скрипт для создания 01_gcn_basics.ipynb
Основы графов и Graph Convolutional Networks
"""

import json

def create_notebook():
    cells = []

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 8: Graph Neural Networks\n",
            "## Часть 1: Основы Графов и GCN\n",
            "\n",
            "### Введение\n",
            "\n",
            "Graph Neural Networks (GNN) - класс нейронных сетей для работы с графовыми данными.\n",
            "Графы позволяют моделировать связи между объектами:\n",
            "\n",
            "- **Социальные сети** - пользователи и их связи\n",
            "- **Молекулы** - атомы и химические связи\n",
            "- **Рекомендательные системы** - пользователи и товары\n",
            "- **Транспортные сети** - станции и маршруты\n",
            "\n",
            "### В этом ноутбуке:\n",
            "\n",
            "1. Основы теории графов\n",
            "2. Представление графов в PyTorch\n",
            "3. Graph Convolutional Network (GCN)\n",
            "4. Message Passing механизм\n",
            "5. Практический пример"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "from torch_geometric.data import Data\n",
            "from torch_geometric.nn import GCNConv, global_mean_pool\n",
            "from torch_geometric.datasets import Planetoid, KarateClub\n",
            "from torch_geometric.utils import to_networkx\n",
            "import networkx as nx\n",
            "from sklearn.manifold import TSNE\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "plt.style.use('seaborn-v0_8-whitegrid')\n",
            "np.random.seed(42)\n",
            "torch.manual_seed(42)\n",
            "\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "print(f'Device: {device}')\n",
            "print(f'PyTorch Geometric loaded')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Основы Теории Графов\n",
            "\n",
            "Граф G = (V, E) состоит из:\n",
            "- **V** - множество вершин (узлов)\n",
            "- **E** - множество рёбер (связей)\n",
            "\n",
            "Каждый узел может иметь признаки (features), каждое ребро - вес."
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Создаём простой граф вручную\n",
            "# 6 узлов, каждый с 3 признаками\n",
            "x = torch.tensor([\n",
            "    [1, 0, 0],  # Node 0\n",
            "    [0, 1, 0],  # Node 1\n",
            "    [0, 0, 1],  # Node 2\n",
            "    [1, 1, 0],  # Node 3\n",
            "    [0, 1, 1],  # Node 4\n",
            "    [1, 0, 1],  # Node 5\n",
            "], dtype=torch.float)\n",
            "\n",
            "# Рёбра в формате COO (source, target)\n",
            "# Граф неориентированный -> добавляем рёбра в обе стороны\n",
            "edge_index = torch.tensor([\n",
            "    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3],  # source\n",
            "    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0],  # target\n",
            "], dtype=torch.long)\n",
            "\n",
            "# Метки узлов (например, для классификации)\n",
            "y = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)\n",
            "\n",
            "# Создаём Data объект PyTorch Geometric\n",
            "data = Data(x=x, edge_index=edge_index, y=y)\n",
            "\n",
            "print('Граф создан:')\n",
            "print(f'  Узлов: {data.num_nodes}')\n",
            "print(f'  Рёбер: {data.num_edges}')\n",
            "print(f'  Признаков на узел: {data.num_node_features}')\n",
            "print(f'  Изолированных узлов: {data.has_isolated_nodes()}')\n",
            "print(f'  Self-loops: {data.has_self_loops()}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Визуализация графа\n",
            "G = to_networkx(data, to_undirected=True)\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(8, 6))\n",
            "\n",
            "# Позиции узлов\n",
            "pos = nx.spring_layout(G, seed=42)\n",
            "\n",
            "# Цвета по меткам\n",
            "colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']\n",
            "node_colors = [colors[label] for label in y.numpy()]\n",
            "\n",
            "nx.draw(G, pos, ax=ax, \n",
            "        node_color=node_colors, \n",
            "        node_size=500,\n",
            "        with_labels=True,\n",
            "        font_size=12,\n",
            "        font_weight='bold',\n",
            "        edge_color='gray',\n",
            "        width=2)\n",
            "\n",
            "ax.set_title('Пример графа с 6 узлами')\n",
            "plt.show()\n",
            "\n",
            "print('Цвета соответствуют классам узлов')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Матрица Смежности\n",
            "\n",
            "Граф можно представить матрицей смежности A:\n",
            "- A[i,j] = 1 если есть ребро между i и j\n",
            "- A[i,j] = 0 иначе"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Создаём матрицу смежности из edge_index\n",
            "num_nodes = data.num_nodes\n",
            "adj_matrix = torch.zeros(num_nodes, num_nodes)\n",
            "\n",
            "for i in range(edge_index.size(1)):\n",
            "    src, dst = edge_index[0, i], edge_index[1, i]\n",
            "    adj_matrix[src, dst] = 1\n",
            "\n",
            "print('Матрица смежности A:')\n",
            "print(adj_matrix.numpy().astype(int))\n",
            "\n",
            "# Степень узла (degree) = сумма по строке\n",
            "degrees = adj_matrix.sum(dim=1)\n",
            "print(f'\\nСтепени узлов: {degrees.numpy().astype(int)}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Graph Convolutional Network (GCN)\n",
            "\n",
            "### Идея GCN\n",
            "\n",
            "GCN обновляет представление узла, агрегируя информацию от соседей:\n",
            "\n",
            "$$H^{(l+1)} = \\sigma(\\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2} H^{(l)} W^{(l)})$$\n",
            "\n",
            "где:\n",
            "- $\\tilde{A} = A + I$ (матрица смежности + self-loops)\n",
            "- $\\tilde{D}$ - диагональная матрица степеней\n",
            "- $H^{(l)}$ - представления узлов на слое l\n",
            "- $W^{(l)}$ - обучаемые веса\n",
            "\n",
            "### Message Passing\n",
            "\n",
            "1. **Aggregate**: собрать сообщения от соседей\n",
            "2. **Update**: обновить представление узла"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Простая реализация GCN слоя\n",
            "class SimpleGCNLayer(nn.Module):\n",
            "    \"\"\"\n",
            "    Упрощённый GCN слой для понимания механизма.\n",
            "    \"\"\"\n",
            "    def __init__(self, in_features, out_features):\n",
            "        super().__init__()\n",
            "        self.linear = nn.Linear(in_features, out_features)\n",
            "    \n",
            "    def forward(self, x, edge_index):\n",
            "        # Добавляем self-loops\n",
            "        num_nodes = x.size(0)\n",
            "        self_loops = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])\n",
            "        edge_index = torch.cat([edge_index, self_loops], dim=1)\n",
            "        \n",
            "        # Вычисляем степени для нормализации\n",
            "        row, col = edge_index\n",
            "        deg = torch.zeros(num_nodes)\n",
            "        deg.scatter_add_(0, row, torch.ones(edge_index.size(1)))\n",
            "        deg_inv_sqrt = deg.pow(-0.5)\n",
            "        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0\n",
            "        \n",
            "        # Нормализация\n",
            "        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]\n",
            "        \n",
            "        # Message passing\n",
            "        # Для каждого узла суммируем нормализованные сообщения от соседей\n",
            "        out = torch.zeros_like(x)\n",
            "        for i in range(edge_index.size(1)):\n",
            "            src, dst = edge_index[0, i], edge_index[1, i]\n",
            "            out[dst] += norm[i] * x[src]\n",
            "        \n",
            "        # Линейное преобразование\n",
            "        out = self.linear(out)\n",
            "        \n",
            "        return out\n",
            "\n",
            "# Тест\n",
            "layer = SimpleGCNLayer(3, 4)\n",
            "out = layer(x, edge_index)\n",
            "print(f'Input shape: {x.shape}')\n",
            "print(f'Output shape: {out.shape}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. GCN с PyTorch Geometric\n",
            "\n",
            "Используем готовую реализацию GCNConv."
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "class GCN(nn.Module):\n",
            "    \"\"\"\n",
            "    Graph Convolutional Network для классификации узлов.\n",
            "    \"\"\"\n",
            "    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):\n",
            "        super().__init__()\n",
            "        self.conv1 = GCNConv(in_channels, hidden_channels)\n",
            "        self.conv2 = GCNConv(hidden_channels, out_channels)\n",
            "        self.dropout = dropout\n",
            "    \n",
            "    def forward(self, x, edge_index):\n",
            "        # Первый GCN слой\n",
            "        x = self.conv1(x, edge_index)\n",
            "        x = F.relu(x)\n",
            "        x = F.dropout(x, p=self.dropout, training=self.training)\n",
            "        \n",
            "        # Второй GCN слой\n",
            "        x = self.conv2(x, edge_index)\n",
            "        \n",
            "        return x\n",
            "    \n",
            "    def get_embeddings(self, x, edge_index):\n",
            "        \"\"\"Получить embeddings после первого слоя.\"\"\"\n",
            "        x = self.conv1(x, edge_index)\n",
            "        x = F.relu(x)\n",
            "        return x\n",
            "\n",
            "print('GCN модель определена')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Датасет Karate Club\n",
            "\n",
            "Классический датасет - социальная сеть карате-клуба Захари.\n",
            "34 члена клуба, связи - взаимодействия вне клуба.\n",
            "Задача: предсказать, к какой группе примкнёт член после раскола."
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Загружаем Karate Club\n",
            "dataset = KarateClub()\n",
            "data = dataset[0]\n",
            "\n",
            "print('Karate Club Dataset:')\n",
            "print(f'  Узлов: {data.num_nodes}')\n",
            "print(f'  Рёбер: {data.num_edges}')\n",
            "print(f'  Признаков: {data.num_node_features}')\n",
            "print(f'  Классов: {dataset.num_classes}')\n",
            "print(f'  Train mask: {data.train_mask.sum().item()} узлов')\n",
            "\n",
            "# Визуализация\n",
            "G = to_networkx(data, to_undirected=True)\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 8))\n",
            "pos = nx.spring_layout(G, seed=42)\n",
            "\n",
            "colors = plt.cm.Set1(data.y.numpy() / data.y.max().item())\n",
            "nx.draw(G, pos, ax=ax,\n",
            "        node_color=colors,\n",
            "        node_size=300,\n",
            "        with_labels=True,\n",
            "        font_size=8)\n",
            "\n",
            "ax.set_title('Zachary\\'s Karate Club')\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Обучение GCN на Karate Club\n",
            "model = GCN(\n",
            "    in_channels=dataset.num_features,\n",
            "    hidden_channels=16,\n",
            "    out_channels=dataset.num_classes,\n",
            "    dropout=0.5\n",
            ")\n",
            "\n",
            "optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)\n",
            "criterion = nn.CrossEntropyLoss()\n",
            "\n",
            "# Training\n",
            "model.train()\n",
            "losses = []\n",
            "\n",
            "for epoch in range(200):\n",
            "    optimizer.zero_grad()\n",
            "    out = model(data.x, data.edge_index)\n",
            "    loss = criterion(out[data.train_mask], data.y[data.train_mask])\n",
            "    loss.backward()\n",
            "    optimizer.step()\n",
            "    losses.append(loss.item())\n",
            "    \n",
            "    if (epoch + 1) % 50 == 0:\n",
            "        model.eval()\n",
            "        pred = out.argmax(dim=1)\n",
            "        acc = (pred == data.y).sum().item() / data.num_nodes\n",
            "        print(f'Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.4f}')\n",
            "        model.train()"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Визуализация результатов\n",
            "model.eval()\n",
            "with torch.no_grad():\n",
            "    embeddings = model.get_embeddings(data.x, data.edge_index)\n",
            "    out = model(data.x, data.edge_index)\n",
            "    pred = out.argmax(dim=1)\n",
            "\n",
            "# t-SNE для визуализации embeddings\n",
            "tsne = TSNE(n_components=2, random_state=42, perplexity=5)\n",
            "embeddings_2d = tsne.fit_transform(embeddings.numpy())\n",
            "\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
            "\n",
            "# 1. Training loss\n",
            "axes[0].plot(losses)\n",
            "axes[0].set_xlabel('Epoch')\n",
            "axes[0].set_ylabel('Loss')\n",
            "axes[0].set_title('Training Loss')\n",
            "\n",
            "# 2. True labels\n",
            "scatter = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], \n",
            "                          c=data.y.numpy(), cmap='Set1', s=100)\n",
            "axes[1].set_title('True Labels (t-SNE)')\n",
            "plt.colorbar(scatter, ax=axes[1])\n",
            "\n",
            "# 3. Predicted labels\n",
            "scatter = axes[2].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],\n",
            "                          c=pred.numpy(), cmap='Set1', s=100)\n",
            "axes[2].set_title('Predicted Labels (t-SNE)')\n",
            "plt.colorbar(scatter, ax=axes[2])\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Accuracy\n",
            "acc = (pred == data.y).sum().item() / data.num_nodes\n",
            "print(f'\\nFinal Accuracy: {acc:.4f}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Cora Dataset - Научные статьи\n",
            "\n",
            "Более сложный датасет:\n",
            "- 2,708 статей (узлы)\n",
            "- 5,429 цитирований (рёбра)\n",
            "- 1,433 признака (bag of words)\n",
            "- 7 классов (тема статьи)"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Загружаем Cora\n",
            "dataset = Planetoid(root='data/Cora', name='Cora')\n",
            "data = dataset[0]\n",
            "\n",
            "print('Cora Dataset:')\n",
            "print(f'  Узлов: {data.num_nodes}')\n",
            "print(f'  Рёбер: {data.num_edges}')\n",
            "print(f'  Признаков: {data.num_node_features}')\n",
            "print(f'  Классов: {dataset.num_classes}')\n",
            "print(f'  Train: {data.train_mask.sum().item()}')\n",
            "print(f'  Val: {data.val_mask.sum().item()}')\n",
            "print(f'  Test: {data.test_mask.sum().item()}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Модель для Cora\n",
            "model = GCN(\n",
            "    in_channels=dataset.num_features,\n",
            "    hidden_channels=64,\n",
            "    out_channels=dataset.num_classes,\n",
            "    dropout=0.5\n",
            ").to(device)\n",
            "\n",
            "data = data.to(device)\n",
            "optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)\n",
            "\n",
            "def train():\n",
            "    model.train()\n",
            "    optimizer.zero_grad()\n",
            "    out = model(data.x, data.edge_index)\n",
            "    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])\n",
            "    loss.backward()\n",
            "    optimizer.step()\n",
            "    return loss.item()\n",
            "\n",
            "def evaluate(mask):\n",
            "    model.eval()\n",
            "    with torch.no_grad():\n",
            "        out = model(data.x, data.edge_index)\n",
            "        pred = out.argmax(dim=1)\n",
            "        correct = (pred[mask] == data.y[mask]).sum().item()\n",
            "        acc = correct / mask.sum().item()\n",
            "    return acc\n",
            "\n",
            "# Training loop\n",
            "best_val_acc = 0\n",
            "train_losses = []\n",
            "\n",
            "for epoch in range(200):\n",
            "    loss = train()\n",
            "    train_losses.append(loss)\n",
            "    \n",
            "    if (epoch + 1) % 20 == 0:\n",
            "        train_acc = evaluate(data.train_mask)\n",
            "        val_acc = evaluate(data.val_mask)\n",
            "        test_acc = evaluate(data.test_mask)\n",
            "        \n",
            "        if val_acc > best_val_acc:\n",
            "            best_val_acc = val_acc\n",
            "            best_test_acc = test_acc\n",
            "        \n",
            "        print(f'Epoch {epoch+1}: loss={loss:.4f}, train={train_acc:.4f}, val={val_acc:.4f}, test={test_acc:.4f}')\n",
            "\n",
            "print(f'\\nBest Test Accuracy: {best_test_acc:.4f}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Итоги\n",
            "\n",
            "### Что мы изучили:\n",
            "\n",
            "1. **Основы графов** - узлы, рёбра, матрица смежности\n",
            "2. **GCN** - свёрточная операция на графах\n",
            "3. **Message Passing** - агрегация информации от соседей\n",
            "4. **Node Classification** - классификация узлов\n",
            "\n",
            "### Ключевые концепции:\n",
            "\n",
            "- GCN учитывает структуру графа\n",
            "- Каждый слой агрегирует информацию от соседей\n",
            "- Несколько слоёв = информация от более далёких узлов\n",
            "- Нормализация важна для стабильности\n",
            "\n",
            "### Следующий шаг:\n",
            "\n",
            "В следующем ноутбуке изучим **Graph Attention Networks (GAT)** - \n",
            "более продвинутую архитектуру с механизмом внимания."
        ]
    })

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "cells": cells
    }
    return notebook

if __name__ == "__main__":
    notebook = create_notebook()
    output_path = "/home/user/test/notebooks/phase8_graph_neural_networks/01_gcn_basics.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"Notebook created: {output_path}")
    print(f"Total cells: {len(notebook['cells'])}")
