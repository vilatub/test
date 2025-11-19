#!/usr/bin/env python3
"""
Скрипт для создания 02_graph_attention.ipynb
Graph Attention Networks (GAT)
"""

import json

def create_notebook():
    cells = []

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 8: Graph Neural Networks\n",
            "## Часть 2: Graph Attention Networks (GAT)\n",
            "\n",
            "### Введение\n",
            "\n",
            "GAT использует механизм внимания для взвешивания соседей:\n",
            "- Не все соседи одинаково важны\n",
            "- Веса внимания обучаются\n",
            "- Интерпретируемость через attention weights\n",
            "\n",
            "### В этом ноутбуке:\n",
            "\n",
            "1. Механизм внимания в графах\n",
            "2. Реализация GAT\n",
            "3. Multi-head attention\n",
            "4. Сравнение с GCN\n",
            "5. Визуализация attention"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "from torch_geometric.nn import GATConv, GCNConv\n",
            "from torch_geometric.datasets import Planetoid\n",
            "from torch_geometric.utils import to_networkx\n",
            "import networkx as nx\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "plt.style.use('seaborn-v0_8-whitegrid')\n",
            "torch.manual_seed(42)\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "print(f'Device: {device}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Механизм Внимания в Графах\n",
            "\n",
            "### GAT Layer\n",
            "\n",
            "Для каждой пары соседних узлов i, j вычисляем attention coefficient:\n",
            "\n",
            "$$e_{ij} = \\text{LeakyReLU}(\\mathbf{a}^T [W\\mathbf{h}_i || W\\mathbf{h}_j])$$\n",
            "\n",
            "Затем нормализуем через softmax:\n",
            "\n",
            "$$\\alpha_{ij} = \\text{softmax}_j(e_{ij}) = \\frac{\\exp(e_{ij})}{\\sum_{k \\in N(i)} \\exp(e_{ik})}$$\n",
            "\n",
            "Обновляем представление узла:\n",
            "\n",
            "$$\\mathbf{h}'_i = \\sigma\\left(\\sum_{j \\in N(i)} \\alpha_{ij} W\\mathbf{h}_j\\right)$$"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Упрощённая реализация GAT слоя\n",
            "class SimpleGATLayer(nn.Module):\n",
            "    \"\"\"\n",
            "    Упрощённый GAT слой для понимания механизма.\n",
            "    \"\"\"\n",
            "    def __init__(self, in_features, out_features):\n",
            "        super().__init__()\n",
            "        self.W = nn.Linear(in_features, out_features, bias=False)\n",
            "        self.a = nn.Linear(2 * out_features, 1, bias=False)\n",
            "        self.leaky_relu = nn.LeakyReLU(0.2)\n",
            "    \n",
            "    def forward(self, x, edge_index):\n",
            "        # Линейное преобразование\n",
            "        Wh = self.W(x)  # [N, out_features]\n",
            "        N = x.size(0)\n",
            "        \n",
            "        # Вычисляем attention coefficients\n",
            "        src, dst = edge_index\n",
            "        \n",
            "        # Конкатенируем представления src и dst\n",
            "        edge_h = torch.cat([Wh[src], Wh[dst]], dim=1)  # [E, 2*out]\n",
            "        e = self.leaky_relu(self.a(edge_h)).squeeze()  # [E]\n",
            "        \n",
            "        # Softmax по соседям каждого узла\n",
            "        attention = torch.zeros(N, N)\n",
            "        attention[src, dst] = e\n",
            "        attention = F.softmax(attention, dim=1)\n",
            "        \n",
            "        # Агрегация\n",
            "        out = torch.matmul(attention, Wh)\n",
            "        \n",
            "        return out, attention\n",
            "\n",
            "print('SimpleGATLayer определён')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. GAT с PyTorch Geometric"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "class GAT(nn.Module):\n",
            "    \"\"\"\n",
            "    Graph Attention Network.\n",
            "    \"\"\"\n",
            "    def __init__(self, in_channels, hidden_channels, out_channels, \n",
            "                 heads=8, dropout=0.6):\n",
            "        super().__init__()\n",
            "        \n",
            "        # Первый слой: multi-head attention\n",
            "        self.conv1 = GATConv(in_channels, hidden_channels, \n",
            "                             heads=heads, dropout=dropout)\n",
            "        \n",
            "        # Второй слой: single head для выхода\n",
            "        self.conv2 = GATConv(hidden_channels * heads, out_channels,\n",
            "                             heads=1, concat=False, dropout=dropout)\n",
            "        \n",
            "        self.dropout = dropout\n",
            "    \n",
            "    def forward(self, x, edge_index):\n",
            "        x = F.dropout(x, p=self.dropout, training=self.training)\n",
            "        x = self.conv1(x, edge_index)\n",
            "        x = F.elu(x)\n",
            "        x = F.dropout(x, p=self.dropout, training=self.training)\n",
            "        x = self.conv2(x, edge_index)\n",
            "        return x\n",
            "\n",
            "print('GAT модель определена')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Загружаем Cora\n",
            "dataset = Planetoid(root='data/Cora', name='Cora')\n",
            "data = dataset[0].to(device)\n",
            "\n",
            "print(f'Cora: {data.num_nodes} узлов, {dataset.num_classes} классов')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Обучение GAT\n",
            "model = GAT(\n",
            "    in_channels=dataset.num_features,\n",
            "    hidden_channels=8,\n",
            "    out_channels=dataset.num_classes,\n",
            "    heads=8,\n",
            "    dropout=0.6\n",
            ").to(device)\n",
            "\n",
            "optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)\n",
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
            "        acc = (pred[mask] == data.y[mask]).sum().item() / mask.sum().item()\n",
            "    return acc\n",
            "\n",
            "# Training\n",
            "best_val_acc = 0\n",
            "gat_losses = []\n",
            "\n",
            "for epoch in range(200):\n",
            "    loss = train()\n",
            "    gat_losses.append(loss)\n",
            "    \n",
            "    if (epoch + 1) % 20 == 0:\n",
            "        val_acc = evaluate(data.val_mask)\n",
            "        test_acc = evaluate(data.test_mask)\n",
            "        \n",
            "        if val_acc > best_val_acc:\n",
            "            best_val_acc = val_acc\n",
            "            best_gat_test = test_acc\n",
            "        \n",
            "        print(f'Epoch {epoch+1}: loss={loss:.4f}, val={val_acc:.4f}, test={test_acc:.4f}')\n",
            "\n",
            "print(f'\\nGAT Best Test: {best_gat_test:.4f}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Сравнение GAT и GCN"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Обучаем GCN для сравнения\n",
            "class GCN(nn.Module):\n",
            "    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):\n",
            "        super().__init__()\n",
            "        self.conv1 = GCNConv(in_channels, hidden_channels)\n",
            "        self.conv2 = GCNConv(hidden_channels, out_channels)\n",
            "        self.dropout = dropout\n",
            "    \n",
            "    def forward(self, x, edge_index):\n",
            "        x = self.conv1(x, edge_index)\n",
            "        x = F.relu(x)\n",
            "        x = F.dropout(x, p=self.dropout, training=self.training)\n",
            "        x = self.conv2(x, edge_index)\n",
            "        return x\n",
            "\n",
            "gcn_model = GCN(\n",
            "    in_channels=dataset.num_features,\n",
            "    hidden_channels=64,\n",
            "    out_channels=dataset.num_classes\n",
            ").to(device)\n",
            "\n",
            "gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)\n",
            "\n",
            "# Training GCN\n",
            "best_gcn_val = 0\n",
            "gcn_losses = []\n",
            "\n",
            "for epoch in range(200):\n",
            "    gcn_model.train()\n",
            "    gcn_optimizer.zero_grad()\n",
            "    out = gcn_model(data.x, data.edge_index)\n",
            "    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])\n",
            "    loss.backward()\n",
            "    gcn_optimizer.step()\n",
            "    gcn_losses.append(loss.item())\n",
            "    \n",
            "    if (epoch + 1) % 200 == 0:\n",
            "        gcn_model.eval()\n",
            "        with torch.no_grad():\n",
            "            out = gcn_model(data.x, data.edge_index)\n",
            "            pred = out.argmax(dim=1)\n",
            "            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()\n",
            "            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()\n",
            "            \n",
            "            if val_acc > best_gcn_val:\n",
            "                best_gcn_val = val_acc\n",
            "                best_gcn_test = test_acc\n",
            "\n",
            "print(f'GCN Best Test: {best_gcn_test:.4f}')\n",
            "print(f'\\nСравнение:')\n",
            "print(f'  GAT: {best_gat_test:.4f}')\n",
            "print(f'  GCN: {best_gcn_test:.4f}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Визуализация\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "\n",
            "axes[0].plot(gat_losses, label='GAT')\n",
            "axes[0].plot(gcn_losses, label='GCN', alpha=0.7)\n",
            "axes[0].set_xlabel('Epoch')\n",
            "axes[0].set_ylabel('Loss')\n",
            "axes[0].set_title('Training Loss')\n",
            "axes[0].legend()\n",
            "\n",
            "# Сравнение accuracy\n",
            "models = ['GCN', 'GAT']\n",
            "accuracies = [best_gcn_test, best_gat_test]\n",
            "colors = ['steelblue', 'coral']\n",
            "axes[1].bar(models, accuracies, color=colors)\n",
            "axes[1].set_ylabel('Test Accuracy')\n",
            "axes[1].set_title('Model Comparison on Cora')\n",
            "axes[1].set_ylim(0.7, 0.9)\n",
            "\n",
            "for i, acc in enumerate(accuracies):\n",
            "    axes[1].text(i, acc + 0.01, f'{acc:.3f}', ha='center')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Multi-Head Attention\n",
            "\n",
            "GAT использует несколько \"голов\" внимания:\n",
            "- Каждая голова учит свои веса\n",
            "- Результаты конкатенируются (или усредняются)\n",
            "- Стабилизирует обучение"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Эксперимент с разным числом голов\n",
            "heads_list = [1, 2, 4, 8]\n",
            "results = []\n",
            "\n",
            "for heads in heads_list:\n",
            "    model = GAT(\n",
            "        in_channels=dataset.num_features,\n",
            "        hidden_channels=8,\n",
            "        out_channels=dataset.num_classes,\n",
            "        heads=heads,\n",
            "        dropout=0.6\n",
            "    ).to(device)\n",
            "    \n",
            "    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)\n",
            "    \n",
            "    best_acc = 0\n",
            "    for epoch in range(200):\n",
            "        model.train()\n",
            "        optimizer.zero_grad()\n",
            "        out = model(data.x, data.edge_index)\n",
            "        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])\n",
            "        loss.backward()\n",
            "        optimizer.step()\n",
            "        \n",
            "        if (epoch + 1) % 50 == 0:\n",
            "            model.eval()\n",
            "            with torch.no_grad():\n",
            "                out = model(data.x, data.edge_index)\n",
            "                pred = out.argmax(dim=1)\n",
            "                acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()\n",
            "                best_acc = max(best_acc, acc)\n",
            "    \n",
            "    results.append({'heads': heads, 'accuracy': best_acc})\n",
            "    print(f'Heads={heads}: {best_acc:.4f}')\n",
            "\n",
            "# Визуализация\n",
            "fig, ax = plt.subplots(figsize=(8, 4))\n",
            "ax.bar([str(r['heads']) for r in results], [r['accuracy'] for r in results])\n",
            "ax.set_xlabel('Number of Attention Heads')\n",
            "ax.set_ylabel('Test Accuracy')\n",
            "ax.set_title('Effect of Multi-Head Attention')\n",
            "plt.show()"
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
            "### GAT vs GCN:\n",
            "\n",
            "| Аспект | GCN | GAT |\n",
            "|--------|-----|-----|\n",
            "| Веса соседей | Фиксированные (по степени) | Обучаемые (attention) |\n",
            "| Интерпретируемость | Ниже | Выше (attention weights) |\n",
            "| Параметры | Меньше | Больше |\n",
            "| Качество | Хорошее | Часто лучше |\n",
            "\n",
            "### Когда использовать GAT:\n",
            "\n",
            "- Важна интерпретируемость\n",
            "- Неоднородная важность соседей\n",
            "- Достаточно данных для обучения attention\n",
            "\n",
            "### Следующий шаг:\n",
            "\n",
            "В ноутбуке 03 применим GNN к практической задаче Node Classification \n",
            "на более сложном датасете."
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
    output_path = "/home/user/test/notebooks/phase8_graph_neural_networks/02_graph_attention.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"Notebook created: {output_path}")
    print(f"Total cells: {len(notebook['cells'])}")
