#!/usr/bin/env python3
"""
Добавление практической части в MLP notebook - PyTorch реализация
"""

import json

# Читаем
notebook_path = '01_mlp_basics.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

practical_cells = []

# ============================================================================
# PRACTICAL: IMPORTS
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📊 Часть 2: Практическая реализация\n",
        "\n",
        "### 2.1 Импорт библиотек"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Основные библиотеки\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# PyTorch\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import Dataset, DataLoader, TensorDataset\n",
        "\n",
        "# Sklearn\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix\n",
        "\n",
        "# Baseline для сравнения\n",
        "from xgboost import XGBClassifier\n",
        "\n",
        "# Визуализация\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "sns.set_palette('husl')\n",
        "%matplotlib inline\n",
        "\n",
        "# Seed\n",
        "RANDOM_STATE = 42\n",
        "np.random.seed(RANDOM_STATE)\n",
        "torch.manual_seed(RANDOM_STATE)\n",
        "\n",
        "# Device\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f'Using device: {device}')\n",
        "print(f'PyTorch version: {torch.__version__}')\n",
        "\n",
        "print('✅ Библиотеки загружены')"
    ]
})

# ============================================================================
# DATA LOADING
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.2 Загрузка и подготовка данных"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Загрузка данных Титаника\n",
        "import os\n",
        "\n",
        "data_path = '../../data/titanic_train.csv'\n",
        "\n",
        "if not os.path.exists(data_path):\n",
        "    print('❌ Файл не найден! Используем альтернативный путь...')\n",
        "    # Попробуем другой путь\n",
        "    data_path = '../titanic/titanic_train.csv'\n",
        "    if not os.path.exists(data_path):\n",
        "        print('❌ Данные не найдены. Создайте файл или загрузите с Kaggle.')\n",
        "else:\n",
        "    df = pd.read_csv(data_path)\n",
        "    print(f'✅ Данные загружены: {df.shape[0]:,} строк, {df.shape[1]} столбцов')\n",
        "    print(f'Target: Survived')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Первый взгляд\n",
        "df.head()"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Подготовка признаков (простая версия для демонстрации)\n",
        "# Выберем основные признаки\n",
        "\n",
        "# Заполняем пропуски\n",
        "df['Age'].fillna(df['Age'].median(), inplace=True)\n",
        "df['Fare'].fillna(df['Fare'].median(), inplace=True)\n",
        "df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)\n",
        "\n",
        "# Создаем признаки\n",
        "df['Sex'] = (df['Sex'] == 'male').astype(int)\n",
        "df['FamilySize'] = df['SibSp'] + df['Parch'] + 1\n",
        "df['IsAlone'] = (df['FamilySize'] == 1).astype(int)\n",
        "\n",
        "# One-hot для Embarked\n",
        "df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)\n",
        "\n",
        "# Выбираем признаки\n",
        "features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone'] + \\\n",
        "           [col for col in df.columns if 'Embarked_' in col]\n",
        "\n",
        "X = df[features].values\n",
        "y = df['Survived'].values\n",
        "\n",
        "print(f'Features: {features}')\n",
        "print(f'X shape: {X.shape}')\n",
        "print(f'y shape: {y.shape}')\n",
        "print(f'Survival rate: {y.mean():.1%}')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Train/Val/Test split\n",
        "X_temp, X_test, y_temp, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y\n",
        ")\n",
        "\n",
        "X_train, X_val, y_train, y_val = train_test_split(\n",
        "    X_temp, y_temp, test_size=0.2, random_state=RANDOM_STATE, stratify=y_temp\n",
        ")\n",
        "\n",
        "print(f'Train: {X_train.shape[0]:,} samples')\n",
        "print(f'Val: {X_val.shape[0]:,} samples')\n",
        "print(f'Test: {X_test.shape[0]:,} samples')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Масштабирование (критично для нейросетей!)\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_val_scaled = scaler.transform(X_val)\n",
        "X_test_scaled = scaler.transform(X_test)\n",
        "\n",
        "print(f'Scaled X_train mean: {X_train_scaled.mean():.4f}')\n",
        "print(f'Scaled X_train std: {X_train_scaled.std():.4f}')\n",
        "print('✅ Данные масштабированы (mean=0, std=1)')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Конвертация в PyTorch tensors\n",
        "X_train_tensor = torch.FloatTensor(X_train_scaled)\n",
        "y_train_tensor = torch.FloatTensor(y_train)\n",
        "\n",
        "X_val_tensor = torch.FloatTensor(X_val_scaled)\n",
        "y_val_tensor = torch.FloatTensor(y_val)\n",
        "\n",
        "X_test_tensor = torch.FloatTensor(X_test_scaled)\n",
        "y_test_tensor = torch.FloatTensor(y_test)\n",
        "\n",
        "print(f'X_train_tensor shape: {X_train_tensor.shape}')\n",
        "print(f'y_train_tensor shape: {y_train_tensor.shape}')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Создаем DataLoaders для mini-batch training\n",
        "batch_size = 32\n",
        "\n",
        "train_dataset = TensorDataset(X_train_tensor, y_train_tensor)\n",
        "val_dataset = TensorDataset(X_val_tensor, y_val_tensor)\n",
        "test_dataset = TensorDataset(X_test_tensor, y_test_tensor)\n",
        "\n",
        "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
        "val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)\n",
        "test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)\n",
        "\n",
        "print(f'Batch size: {batch_size}')\n",
        "print(f'Number of batches (train): {len(train_loader)}')\n",
        "print(f'Number of batches (val): {len(val_loader)}')"
    ]
})

# ============================================================================
# MLP ARCHITECTURE
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.3 Определение MLP архитектуры"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Простая MLP архитектура\n",
        "class SimpleMLP(nn.Module):\n",
        "    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.3):\n",
        "        super(SimpleMLP, self).__init__()\n",
        "        \n",
        "        self.fc1 = nn.Linear(input_dim, hidden_dim1)\n",
        "        self.bn1 = nn.BatchNorm1d(hidden_dim1)\n",
        "        self.dropout1 = nn.Dropout(dropout_rate)\n",
        "        \n",
        "        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)\n",
        "        self.bn2 = nn.BatchNorm1d(hidden_dim2)\n",
        "        self.dropout2 = nn.Dropout(dropout_rate)\n",
        "        \n",
        "        self.fc3 = nn.Linear(hidden_dim2, 1)\n",
        "    \n",
        "    def forward(self, x):\n",
        "        # Layer 1: Linear → BatchNorm → ReLU → Dropout\n",
        "        x = self.fc1(x)\n",
        "        x = self.bn1(x)\n",
        "        x = torch.relu(x)\n",
        "        x = self.dropout1(x)\n",
        "        \n",
        "        # Layer 2: Linear → BatchNorm → ReLU → Dropout\n",
        "        x = self.fc2(x)\n",
        "        x = self.bn2(x)\n",
        "        x = torch.relu(x)\n",
        "        x = self.dropout2(x)\n",
        "        \n",
        "        # Output: Linear → Sigmoid\n",
        "        x = self.fc3(x)\n",
        "        x = torch.sigmoid(x)\n",
        "        \n",
        "        return x\n",
        "\n",
        "# Инициализация модели\n",
        "input_dim = X_train_scaled.shape[1]\n",
        "model = SimpleMLP(input_dim=input_dim)\n",
        "model = model.to(device)\n",
        "\n",
        "print(model)\n",
        "print(f'\\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}')\n",
        "print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')"
    ]
})

# ============================================================================
# TRAINING
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.4 Обучение модели"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Loss и optimizer\n",
        "criterion = nn.BCELoss()  # Binary Cross-Entropy\n",
        "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
        "\n",
        "print(f'Loss function: {criterion}')\n",
        "print(f'Optimizer: {optimizer}')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Training функция\n",
        "def train_epoch(model, loader, criterion, optimizer, device):\n",
        "    model.train()\n",
        "    total_loss = 0\n",
        "    correct = 0\n",
        "    total = 0\n",
        "    \n",
        "    for X_batch, y_batch in loader:\n",
        "        X_batch = X_batch.to(device)\n",
        "        y_batch = y_batch.to(device).unsqueeze(1)  # (batch_size, 1)\n",
        "        \n",
        "        # Forward pass\n",
        "        outputs = model(X_batch)\n",
        "        loss = criterion(outputs, y_batch)\n",
        "        \n",
        "        # Backward pass\n",
        "        optimizer.zero_grad()\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        \n",
        "        # Метрики\n",
        "        total_loss += loss.item() * X_batch.size(0)\n",
        "        predicted = (outputs > 0.5).float()\n",
        "        correct += (predicted == y_batch).sum().item()\n",
        "        total += y_batch.size(0)\n",
        "    \n",
        "    avg_loss = total_loss / total\n",
        "    accuracy = correct / total\n",
        "    return avg_loss, accuracy\n",
        "\n",
        "# Validation функция\n",
        "def validate_epoch(model, loader, criterion, device):\n",
        "    model.eval()\n",
        "    total_loss = 0\n",
        "    correct = 0\n",
        "    total = 0\n",
        "    all_preds = []\n",
        "    all_labels = []\n",
        "    \n",
        "    with torch.no_grad():\n",
        "        for X_batch, y_batch in loader:\n",
        "            X_batch = X_batch.to(device)\n",
        "            y_batch = y_batch.to(device).unsqueeze(1)\n",
        "            \n",
        "            outputs = model(X_batch)\n",
        "            loss = criterion(outputs, y_batch)\n",
        "            \n",
        "            total_loss += loss.item() * X_batch.size(0)\n",
        "            predicted = (outputs > 0.5).float()\n",
        "            correct += (predicted == y_batch).sum().item()\n",
        "            total += y_batch.size(0)\n",
        "            \n",
        "            all_preds.extend(outputs.cpu().numpy())\n",
        "            all_labels.extend(y_batch.cpu().numpy())\n",
        "    \n",
        "    avg_loss = total_loss / total\n",
        "    accuracy = correct / total\n",
        "    return avg_loss, accuracy, all_preds, all_labels\n",
        "\n",
        "print('✅ Training functions готовы')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Training loop\n",
        "num_epochs = 100\n",
        "patience = 10\n",
        "\n",
        "train_losses = []\n",
        "val_losses = []\n",
        "train_accs = []\n",
        "val_accs = []\n",
        "\n",
        "best_val_loss = float('inf')\n",
        "patience_counter = 0\n",
        "best_model_state = None\n",
        "\n",
        "print('Начинаем обучение...')\n",
        "print('='*60)\n",
        "\n",
        "for epoch in range(num_epochs):\n",
        "    # Train\n",
        "    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)\n",
        "    \n",
        "    # Validate\n",
        "    val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)\n",
        "    \n",
        "    # Сохраняем метрики\n",
        "    train_losses.append(train_loss)\n",
        "    val_losses.append(val_loss)\n",
        "    train_accs.append(train_acc)\n",
        "    val_accs.append(val_acc)\n",
        "    \n",
        "    # Early stopping\n",
        "    if val_loss < best_val_loss:\n",
        "        best_val_loss = val_loss\n",
        "        best_model_state = model.state_dict().copy()\n",
        "        patience_counter = 0\n",
        "    else:\n",
        "        patience_counter += 1\n",
        "    \n",
        "    # Print каждые 10 эпох\n",
        "    if (epoch + 1) % 10 == 0:\n",
        "        print(f'Epoch [{epoch+1}/{num_epochs}]')\n",
        "        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')\n",
        "        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')\n",
        "        print(f'  Best Val Loss: {best_val_loss:.4f}, Patience: {patience_counter}/{patience}')\n",
        "    \n",
        "    # Early stopping\n",
        "    if patience_counter >= patience:\n",
        "        print(f'\\nEarly stopping на epoch {epoch+1}')\n",
        "        break\n",
        "\n",
        "# Загружаем лучшую модель\n",
        "model.load_state_dict(best_model_state)\n",
        "\n",
        "print('='*60)\n",
        "print(f'✅ Обучение завершено!')\n",
        "print(f'Best validation loss: {best_val_loss:.4f}')\n",
        "print(f'Total epochs: {epoch+1}')"
    ]
})

# ============================================================================
# VISUALIZATION
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.5 Визуализация обучения"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# График обучения\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# Loss\n",
        "ax1.plot(train_losses, label='Train Loss', alpha=0.8)\n",
        "ax1.plot(val_losses, label='Val Loss', alpha=0.8)\n",
        "ax1.set_xlabel('Epoch')\n",
        "ax1.set_ylabel('Loss (BCE)')\n",
        "ax1.set_title('Training and Validation Loss')\n",
        "ax1.legend()\n",
        "ax1.grid(True, alpha=0.3)\n",
        "\n",
        "# Accuracy\n",
        "ax2.plot(train_accs, label='Train Accuracy', alpha=0.8)\n",
        "ax2.plot(val_accs, label='Val Accuracy', alpha=0.8)\n",
        "ax2.set_xlabel('Epoch')\n",
        "ax2.set_ylabel('Accuracy')\n",
        "ax2.set_title('Training and Validation Accuracy')\n",
        "ax2.legend()\n",
        "ax2.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(f'Final train accuracy: {train_accs[-1]:.4f}')\n",
        "print(f'Final val accuracy: {val_accs[-1]:.4f}')\n",
        "print(f'Overfitting: {(train_accs[-1] - val_accs[-1]):.4f}')"
    ]
})

# ============================================================================
# EVALUATION
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.6 Оценка на test set"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Оценка на test\n",
        "test_loss, test_acc, test_preds, test_labels = validate_epoch(\n",
        "    model, test_loader, criterion, device\n",
        ")\n",
        "\n",
        "# Конвертация в numpy\n",
        "test_preds_np = np.array(test_preds).flatten()\n",
        "test_labels_np = np.array(test_labels).flatten()\n",
        "test_preds_binary = (test_preds_np > 0.5).astype(int)\n",
        "\n",
        "# Метрики\n",
        "test_precision = precision_score(test_labels_np, test_preds_binary)\n",
        "test_recall = recall_score(test_labels_np, test_preds_binary)\n",
        "test_f1 = f1_score(test_labels_np, test_preds_binary)\n",
        "test_roc_auc = roc_auc_score(test_labels_np, test_preds_np)\n",
        "\n",
        "print('📊 MLP Test Results:')\n",
        "print('='*50)\n",
        "print(f'  Test Loss: {test_loss:.4f}')\n",
        "print(f'  Test Accuracy: {test_acc:.4f}')\n",
        "print(f'  Precision: {test_precision:.4f}')\n",
        "print(f'  Recall: {test_recall:.4f}')\n",
        "print(f'  F1-score: {test_f1:.4f}')\n",
        "print(f'  ROC-AUC: {test_roc_auc:.4f}')\n",
        "print('='*50)"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Confusion Matrix\n",
        "cm = confusion_matrix(test_labels_np, test_preds_binary)\n",
        "\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)\n",
        "plt.xlabel('Predicted')\n",
        "plt.ylabel('Actual')\n",
        "plt.title('MLP Confusion Matrix (Test Set)')\n",
        "plt.show()\n",
        "\n",
        "print(f'True Negatives: {cm[0,0]}')\n",
        "print(f'False Positives: {cm[0,1]}')\n",
        "print(f'False Negatives: {cm[1,0]}')\n",
        "print(f'True Positives: {cm[1,1]}')"
    ]
})

# ============================================================================
# COMPARISON WITH XGBOOST
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.7 Сравнение с XGBoost (Phase 1)"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# XGBoost для сравнения\n",
        "print('Обучаем XGBoost для честного сравнения...')\n",
        "\n",
        "xgb_model = XGBClassifier(\n",
        "    n_estimators=100,\n",
        "    learning_rate=0.1,\n",
        "    max_depth=4,\n",
        "    random_state=RANDOM_STATE,\n",
        "    verbosity=0\n",
        ")\n",
        "\n",
        "xgb_model.fit(X_train_scaled, y_train)\n",
        "\n",
        "# Предсказания XGBoost\n",
        "xgb_preds = xgb_model.predict(X_test_scaled)\n",
        "xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]\n",
        "\n",
        "# Метрики XGBoost\n",
        "xgb_acc = accuracy_score(y_test, xgb_preds)\n",
        "xgb_precision = precision_score(y_test, xgb_preds)\n",
        "xgb_recall = recall_score(y_test, xgb_preds)\n",
        "xgb_f1 = f1_score(y_test, xgb_preds)\n",
        "xgb_roc_auc = roc_auc_score(y_test, xgb_proba)\n",
        "\n",
        "print('✅ XGBoost обучен')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Сравнительная таблица\n",
        "comparison = pd.DataFrame({\n",
        "    'Model': ['MLP (PyTorch)', 'XGBoost'],\n",
        "    'Accuracy': [test_acc, xgb_acc],\n",
        "    'Precision': [test_precision, xgb_precision],\n",
        "    'Recall': [test_recall, xgb_recall],\n",
        "    'F1-score': [test_f1, xgb_f1],\n",
        "    'ROC-AUC': [test_roc_auc, xgb_roc_auc]\n",
        "})\n",
        "\n",
        "print('\\n' + '='*70)\n",
        "print('🏆 СРАВНЕНИЕ: MLP vs XGBoost')\n",
        "print('='*70)\n",
        "print(comparison.to_string(index=False))\n",
        "print('='*70)\n",
        "\n",
        "# Визуализация\n",
        "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n",
        "\n",
        "metrics = ['Accuracy', 'F1-score', 'ROC-AUC']\n",
        "for i, metric in enumerate(metrics):\n",
        "    axes[i].bar(['MLP', 'XGBoost'], comparison[metric], alpha=0.7, edgecolor='black')\n",
        "    axes[i].set_ylabel(metric)\n",
        "    axes[i].set_title(f'{metric} Comparison')\n",
        "    axes[i].set_ylim([0.7, 0.9])\n",
        "    axes[i].grid(True, alpha=0.3, axis='y')\n",
        "    \n",
        "    # Добавляем значения\n",
        "    for j, v in enumerate(comparison[metric]):\n",
        "        axes[i].text(j, v + 0.01, f'{v:.4f}', ha='center')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# ============================================================================
# CONCLUSIONS
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🎯 Выводы\n",
        "\n",
        "### Что мы изучили:\n",
        "\n",
        "1. **MLP архитектура:**\n",
        "   - Полносвязные слои (Linear)\n",
        "   - Функции активации (ReLU, Sigmoid)\n",
        "   - Batch Normalization\n",
        "   - Dropout для регуляризации\n",
        "\n",
        "2. **Обучение нейросетей:**\n",
        "   - Backpropagation и gradient descent\n",
        "   - Adam optimizer\n",
        "   - Binary Cross-Entropy loss\n",
        "   - Early stopping\n",
        "   - Mini-batch training\n",
        "\n",
        "3. **PyTorch основы:**\n",
        "   - `nn.Module` для определения модели\n",
        "   - `DataLoader` для batch processing\n",
        "   - Training loop (forward → loss → backward → update)\n",
        "   - GPU support\n",
        "\n",
        "### MLP vs XGBoost для табличных данных:\n",
        "\n",
        "**Типичные результаты:**\n",
        "- MLP: Accuracy ~78-82%, ROC-AUC ~0.82-0.85\n",
        "- XGBoost: Accuracy ~80-84%, ROC-AUC ~0.84-0.87\n",
        "\n",
        "**Вывод:** XGBoost обычно **лучше** для табличных данных!\n",
        "\n",
        "### Когда использовать MLP?\n",
        "\n",
        "✅ **Используйте MLP когда:**\n",
        "- Много данных (>100k примеров)\n",
        "- Сложные нелинейные зависимости\n",
        "- Нужен ensemble с Gradient Boosting\n",
        "- Transfer learning (предобучение на смежной задаче)\n",
        "- Изучение Deep Learning основ\n",
        "\n",
        "❌ **НЕ используйте MLP когда:**\n",
        "- Мало данных (<10k)\n",
        "- Нужна интерпретируемость\n",
        "- Ограниченные ресурсы (XGBoost быстрее)\n",
        "- Production система (XGBoost проще deploy)\n",
        "\n",
        "### Ключевые уроки:\n",
        "\n",
        "1. **Scaling критичен!** Нейросети требуют нормализации (mean=0, std=1)\n",
        "2. **Batch Normalization** ускоряет обучение\n",
        "3. **Dropout** предотвращает overfitting\n",
        "4. **Early stopping** экономит время и улучшает обобщение\n",
        "5. **Learning rate** - критичный гиперпараметр\n",
        "6. **Adam optimizer** работает хорошо out-of-the-box\n",
        "\n",
        "### Следующие шаги:\n",
        "\n",
        "1. **Эксперименты:**\n",
        "   - Разные архитектуры (глубже, шире)\n",
        "   - Другие функции активации (ELU, LeakyReLU)\n",
        "   - Learning rate scheduling\n",
        "   - Разные optimizers (SGD, AdamW)\n",
        "\n",
        "2. **1D-CNN:** Convolutional layers для табличных данных\n",
        "3. **Autoencoders:** Unsupervised learning, anomaly detection\n",
        "4. **Transfer Learning:** Предобучение на других данных\n",
        "\n",
        "---\n",
        "\n",
        "## 🎉 Phase 2, Step 1 завершен!\n",
        "\n",
        "Вы освоили основы Deep Learning с MLP. Теперь готовы к более сложным архитектурам! 🚀\n"
    ]
})

# Добавляем практические ячейки
for cell in practical_cells:
    notebook['cells'].append(cell)

# Сохраняем
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Добавлена практика: {len(practical_cells)} ячеек')
print(f'Всего ячеек в ноутбуке: {len(notebook["cells"])}')
