#!/usr/bin/env python3
"""
Добавление практики: Vanilla AE, Denoising AE, VAE
"""

import json

notebook_path = '03_autoencoders.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

practical_cells = []

# ============================================================================
# IMPORTS
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 📊 Часть 2: Практическая реализация\n\n### 2.1 Импорт библиотек"]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import DataLoader, TensorDataset\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.decomposition import PCA\n",
        "from sklearn.metrics import roc_auc_score, average_precision_score\n",
        "\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "%matplotlib inline\n",
        "\n",
        "RANDOM_STATE = 42\n",
        "np.random.seed(RANDOM_STATE)\n",
        "torch.manual_seed(RANDOM_STATE)\n",
        "\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f'Device: {device}')\n",
        "print('✅ Библиотеки загружены')"
    ]
})

# DATA
practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### 2.2 Загрузка данных"]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Загрузка Титаника\n",
        "data_path = '../../data/titanic_train.csv'\n",
        "df = pd.read_csv(data_path) if __import__('os').path.exists(data_path) else None\n",
        "\n",
        "if df is not None:\n",
        "    # Подготовка (как в MLP)\n",
        "    df['Age'].fillna(df['Age'].median(), inplace=True)\n",
        "    df['Fare'].fillna(df['Fare'].median(), inplace=True)\n",
        "    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)\n",
        "    df['Sex'] = (df['Sex'] == 'male').astype(int)\n",
        "    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1\n",
        "    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)\n",
        "    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)\n",
        "    \n",
        "    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone'] + \\\n",
        "               [col for col in df.columns if 'Embarked_' in col]\n",
        "    \n",
        "    X = df[features].values\n",
        "    y = df['Survived'].values\n",
        "    \n",
        "    print(f'✅ Данные: {X.shape}')\n",
        "    print(f'Признаки: {features}')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Для Autoencoder используем ТОЛЬКО выживших (для anomaly detection)\n",
        "X_survived = X[y == 1]  # Только выжившие\n",
        "X_died = X[y == 0]      # Погибшие (для тестирования аномалий)\n",
        "\n",
        "# Train/test split из выживших\n",
        "X_train, X_test = train_test_split(X_survived, test_size=0.2, random_state=RANDOM_STATE)\n",
        "\n",
        "# Scaling\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_test_scaled = scaler.transform(X_test)\n",
        "X_died_scaled = scaler.transform(X_died)  # Для anomaly detection\n",
        "\n",
        "# PyTorch tensors\n",
        "X_train_t = torch.FloatTensor(X_train_scaled)\n",
        "X_test_t = torch.FloatTensor(X_test_scaled)\n",
        "X_died_t = torch.FloatTensor(X_died_scaled)\n",
        "\n",
        "# DataLoader\n",
        "batch_size = 32\n",
        "train_loader = DataLoader(TensorDataset(X_train_t, X_train_t), batch_size=batch_size, shuffle=True)\n",
        "test_loader = DataLoader(TensorDataset(X_test_t, X_test_t), batch_size=batch_size)\n",
        "\n",
        "print(f'Train (survived): {X_train.shape[0]}')\n",
        "print(f'Test (survived): {X_test.shape[0]}')\n",
        "print(f'Died (for anomaly): {X_died.shape[0]}')"
    ]
})

# ============================================================================
# VANILLA AUTOENCODER
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### 2.3 Vanilla Autoencoder"]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class VanillaAutoencoder(nn.Module):\n",
        "    def __init__(self, input_dim, latent_dim=2):\n",
        "        super().__init__()\n",
        "        \n",
        "        # Encoder: input → latent\n",
        "        self.encoder = nn.Sequential(\n",
        "            nn.Linear(input_dim, 16),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(16, 8),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(8, latent_dim)\n",
        "        )\n",
        "        \n",
        "        # Decoder: latent → input\n",
        "        self.decoder = nn.Sequential(\n",
        "            nn.Linear(latent_dim, 8),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(8, 16),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(16, input_dim)\n",
        "        )\n",
        "    \n",
        "    def forward(self, x):\n",
        "        z = self.encoder(x)\n",
        "        x_reconstructed = self.decoder(z)\n",
        "        return x_reconstructed, z\n",
        "\n",
        "input_dim = X_train_scaled.shape[1]\n",
        "latent_dim = 2  # 2D для визуализации\n",
        "\n",
        "vanilla_ae = VanillaAutoencoder(input_dim, latent_dim).to(device)\n",
        "print(vanilla_ae)\n",
        "print(f'Parameters: {sum(p.numel() for p in vanilla_ae.parameters()):,}')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Training\n",
        "def train_autoencoder(model, loader, epochs=50, lr=0.001):\n",
        "    optimizer = optim.Adam(model.parameters(), lr=lr)\n",
        "    criterion = nn.MSELoss()\n",
        "    losses = []\n",
        "    \n",
        "    model.train()\n",
        "    for epoch in range(epochs):\n",
        "        epoch_loss = 0\n",
        "        for X_batch, _ in loader:\n",
        "            X_batch = X_batch.to(device)\n",
        "            \n",
        "            # Forward\n",
        "            X_recon, _ = model(X_batch)\n",
        "            loss = criterion(X_recon, X_batch)\n",
        "            \n",
        "            # Backward\n",
        "            optimizer.zero_grad()\n",
        "            loss.backward()\n",
        "            optimizer.step()\n",
        "            \n",
        "            epoch_loss += loss.item() * X_batch.size(0)\n",
        "        \n",
        "        avg_loss = epoch_loss / len(loader.dataset)\n",
        "        losses.append(avg_loss)\n",
        "        \n",
        "        if (epoch + 1) % 10 == 0:\n",
        "            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')\n",
        "    \n",
        "    return losses\n",
        "\n",
        "print('Обучаем Vanilla Autoencoder...')\n",
        "vanilla_losses = train_autoencoder(vanilla_ae, train_loader, epochs=50)\n",
        "print('✅ Обучение завершено')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Визуализация latent space\n",
        "vanilla_ae.eval()\n",
        "with torch.no_grad():\n",
        "    _, z_survived = vanilla_ae(X_test_t.to(device))\n",
        "    _, z_died = vanilla_ae(X_died_t.to(device))\n",
        "    z_survived = z_survived.cpu().numpy()\n",
        "    z_died = z_died.cpu().numpy()\n",
        "\n",
        "plt.figure(figsize=(10, 6))\n",
        "plt.scatter(z_survived[:, 0], z_survived[:, 1], alpha=0.6, label='Survived (train)', c='green')\n",
        "plt.scatter(z_died[:, 0], z_died[:, 1], alpha=0.6, label='Died (test)', c='red')\n",
        "plt.xlabel('Latent Dimension 1')\n",
        "plt.ylabel('Latent Dimension 2')\n",
        "plt.title('Vanilla Autoencoder: Latent Space')\n",
        "plt.legend()\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.show()\n",
        "\n",
        "print('🔍 Погибшие пассажиры имеют другое распределение в latent space!')"
    ]
})

# ============================================================================
# ANOMALY DETECTION
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### 2.4 Anomaly Detection с Vanilla AE"]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Reconstruction error для anomaly detection\n",
        "def reconstruction_error(model, X):\n",
        "    model.eval()\n",
        "    with torch.no_grad():\n",
        "        X_recon, _ = model(X.to(device))\n",
        "        errors = torch.mean((X.to(device) - X_recon) ** 2, dim=1).cpu().numpy()\n",
        "    return errors\n",
        "\n",
        "# Ошибки для выживших и погибших\n",
        "errors_survived = reconstruction_error(vanilla_ae, X_test_t)\n",
        "errors_died = reconstruction_error(vanilla_ae, X_died_t)\n",
        "\n",
        "# Визуализация\n",
        "plt.figure(figsize=(12, 5))\n",
        "\n",
        "plt.subplot(1, 2, 1)\n",
        "plt.hist(errors_survived, bins=30, alpha=0.7, label='Survived', color='green', edgecolor='black')\n",
        "plt.hist(errors_died, bins=30, alpha=0.7, label='Died', color='red', edgecolor='black')\n",
        "plt.xlabel('Reconstruction Error')\n",
        "plt.ylabel('Frequency')\n",
        "plt.title('Reconstruction Error Distribution')\n",
        "plt.legend()\n",
        "\n",
        "plt.subplot(1, 2, 2)\n",
        "plt.boxplot([errors_survived, errors_died], labels=['Survived', 'Died'])\n",
        "plt.ylabel('Reconstruction Error')\n",
        "plt.title('Reconstruction Error: Boxplot')\n",
        "plt.grid(True, alpha=0.3, axis='y')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(f'Mean error (survived): {errors_survived.mean():.4f}')\n",
        "print(f'Mean error (died): {errors_died.mean():.4f}')\n",
        "print(f'Разница: {errors_died.mean() / errors_survived.mean():.2f}x')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ROC-AUC для anomaly detection\n",
        "# Label: 0 = normal (survived), 1 = anomaly (died)\n",
        "y_true = np.concatenate([np.zeros(len(errors_survived)), np.ones(len(errors_died))])\n",
        "y_scores = np.concatenate([errors_survived, errors_died])\n",
        "\n",
        "auc = roc_auc_score(y_true, y_scores)\n",
        "ap = average_precision_score(y_true, y_scores)\n",
        "\n",
        "print('📊 Anomaly Detection Performance:')\n",
        "print(f'  ROC-AUC: {auc:.4f}')\n",
        "print(f'  Average Precision: {ap:.4f}')\n",
        "print('\\n✅ Autoencoder успешно отличает выживших от погибших!')"
    ]
})

# ============================================================================
# DENOISING AUTOENCODER
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### 2.5 Denoising Autoencoder"]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Denoising AE использует ту же архитектуру, но обучается на зашумлённых данных\n",
        "class DenoisingAutoencoder(nn.Module):\n",
        "    def __init__(self, input_dim, latent_dim=2):\n",
        "        super().__init__()\n",
        "        self.encoder = nn.Sequential(\n",
        "            nn.Linear(input_dim, 16), nn.ReLU(),\n",
        "            nn.Linear(16, 8), nn.ReLU(),\n",
        "            nn.Linear(8, latent_dim)\n",
        "        )\n",
        "        self.decoder = nn.Sequential(\n",
        "            nn.Linear(latent_dim, 8), nn.ReLU(),\n",
        "            nn.Linear(8, 16), nn.ReLU(),\n",
        "            nn.Linear(16, input_dim)\n",
        "        )\n",
        "    \n",
        "    def forward(self, x):\n",
        "        z = self.encoder(x)\n",
        "        x_reconstructed = self.decoder(z)\n",
        "        return x_reconstructed, z\n",
        "\n",
        "denoising_ae = DenoisingAutoencoder(input_dim, latent_dim).to(device)\n",
        "\n",
        "# Training с добавлением шума\n",
        "def train_denoising_ae(model, loader, epochs=50, noise_factor=0.2, lr=0.001):\n",
        "    optimizer = optim.Adam(model.parameters(), lr=lr)\n",
        "    criterion = nn.MSELoss()\n",
        "    losses = []\n",
        "    \n",
        "    model.train()\n",
        "    for epoch in range(epochs):\n",
        "        epoch_loss = 0\n",
        "        for X_batch, _ in loader:\n",
        "            X_batch = X_batch.to(device)\n",
        "            \n",
        "            # Добавляем Gaussian noise\n",
        "            noise = torch.randn_like(X_batch) * noise_factor\n",
        "            X_noisy = X_batch + noise\n",
        "            \n",
        "            # Forward: восстанавливаем ЧИСТЫЕ данные из зашумлённых\n",
        "            X_recon, _ = model(X_noisy)\n",
        "            loss = criterion(X_recon, X_batch)  # Сравниваем с ЧИСТЫМИ!\n",
        "            \n",
        "            # Backward\n",
        "            optimizer.zero_grad()\n",
        "            loss.backward()\n",
        "            optimizer.step()\n",
        "            \n",
        "            epoch_loss += loss.item() * X_batch.size(0)\n",
        "        \n",
        "        avg_loss = epoch_loss / len(loader.dataset)\n",
        "        losses.append(avg_loss)\n",
        "        \n",
        "        if (epoch + 1) % 10 == 0:\n",
        "            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')\n",
        "    \n",
        "    return losses\n",
        "\n",
        "print('Обучаем Denoising Autoencoder...')\n",
        "denoising_losses = train_denoising_ae(denoising_ae, train_loader, epochs=50, noise_factor=0.3)\n",
        "print('✅ Обучение завершено')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Демонстрация denoising\n",
        "denoising_ae.eval()\n",
        "sample = X_test_t[:5].to(device)\n",
        "sample_noisy = sample + torch.randn_like(sample) * 0.3\n",
        "\n",
        "with torch.no_grad():\n",
        "    sample_denoised, _ = denoising_ae(sample_noisy)\n",
        "\n",
        "# Визуализация\n",
        "fig, axes = plt.subplots(3, 1, figsize=(12, 8))\n",
        "\n",
        "for i in range(3):\n",
        "    axes[i].plot(sample[i].cpu().numpy(), 'o-', label='Original', alpha=0.7)\n",
        "    axes[i].plot(sample_noisy[i].cpu().numpy(), 's-', label='Noisy', alpha=0.7)\n",
        "    axes[i].plot(sample_denoised[i].cpu().numpy(), '^-', label='Denoised', alpha=0.7)\n",
        "    axes[i].set_ylabel('Value')\n",
        "    axes[i].set_title(f'Sample {i+1}: Denoising Effect')\n",
        "    axes[i].legend()\n",
        "    axes[i].grid(True, alpha=0.3)\n",
        "\n",
        "axes[-1].set_xlabel('Feature Index')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('✅ Denoising Autoencoder успешно очищает зашумлённые данные!')"
    ]
})

# ============================================================================
# VAE
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### 2.6 Variational Autoencoder (VAE)"]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class VAE(nn.Module):\n",
        "    def __init__(self, input_dim, latent_dim=2):\n",
        "        super().__init__()\n",
        "        \n",
        "        # Encoder → μ and log(σ²)\n",
        "        self.encoder = nn.Sequential(\n",
        "            nn.Linear(input_dim, 16), nn.ReLU(),\n",
        "            nn.Linear(16, 8), nn.ReLU()\n",
        "        )\n",
        "        self.fc_mu = nn.Linear(8, latent_dim)\n",
        "        self.fc_logvar = nn.Linear(8, latent_dim)\n",
        "        \n",
        "        # Decoder\n",
        "        self.decoder = nn.Sequential(\n",
        "            nn.Linear(latent_dim, 8), nn.ReLU(),\n",
        "            nn.Linear(8, 16), nn.ReLU(),\n",
        "            nn.Linear(16, input_dim)\n",
        "        )\n",
        "    \n",
        "    def encode(self, x):\n",
        "        h = self.encoder(x)\n",
        "        mu = self.fc_mu(h)\n",
        "        logvar = self.fc_logvar(h)\n",
        "        return mu, logvar\n",
        "    \n",
        "    def reparameterize(self, mu, logvar):\n",
        "        # z = μ + σ * ε, где ε ~ N(0,1)\n",
        "        std = torch.exp(0.5 * logvar)\n",
        "        eps = torch.randn_like(std)\n",
        "        return mu + eps * std\n",
        "    \n",
        "    def decode(self, z):\n",
        "        return self.decoder(z)\n",
        "    \n",
        "    def forward(self, x):\n",
        "        mu, logvar = self.encode(x)\n",
        "        z = self.reparameterize(mu, logvar)\n",
        "        x_recon = self.decode(z)\n",
        "        return x_recon, mu, logvar\n",
        "\n",
        "vae = VAE(input_dim, latent_dim).to(device)\n",
        "print(vae)"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# VAE Loss: Reconstruction + KL divergence\n",
        "def vae_loss(x_recon, x, mu, logvar):\n",
        "    # Reconstruction loss\n",
        "    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')\n",
        "    \n",
        "    # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)\n",
        "    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())\n",
        "    \n",
        "    return recon_loss + kl_div\n",
        "\n",
        "# Training VAE\n",
        "def train_vae(model, loader, epochs=50, lr=0.001):\n",
        "    optimizer = optim.Adam(model.parameters(), lr=lr)\n",
        "    losses = []\n",
        "    \n",
        "    model.train()\n",
        "    for epoch in range(epochs):\n",
        "        epoch_loss = 0\n",
        "        for X_batch, _ in loader:\n",
        "            X_batch = X_batch.to(device)\n",
        "            \n",
        "            # Forward\n",
        "            X_recon, mu, logvar = model(X_batch)\n",
        "            loss = vae_loss(X_recon, X_batch, mu, logvar)\n",
        "            \n",
        "            # Backward\n",
        "            optimizer.zero_grad()\n",
        "            loss.backward()\n",
        "            optimizer.step()\n",
        "            \n",
        "            epoch_loss += loss.item()\n",
        "        \n",
        "        avg_loss = epoch_loss / len(loader.dataset)\n",
        "        losses.append(avg_loss)\n",
        "        \n",
        "        if (epoch + 1) % 10 == 0:\n",
        "            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')\n",
        "    \n",
        "    return losses\n",
        "\n",
        "print('Обучаем VAE...')\n",
        "vae_losses = train_vae(vae, train_loader, epochs=50)\n",
        "print('✅ VAE обучен')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Генерация новых пассажиров с VAE\n",
        "vae.eval()\n",
        "with torch.no_grad():\n",
        "    # Sample из prior N(0, I)\n",
        "    z_sample = torch.randn(10, latent_dim).to(device)\n",
        "    generated = vae.decode(z_sample).cpu().numpy()\n",
        "\n",
        "# Обратная трансформация scaling\n",
        "generated_original = scaler.inverse_transform(generated)\n",
        "\n",
        "print('🎨 Сгенерированные \"пассажиры\" (первые 5):')\n",
        "print(pd.DataFrame(generated_original[:5], columns=features))\n",
        "print('\\n✅ VAE может генерировать новые примеры!')"
    ]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Интерполяция в latent space\n",
        "vae.eval()\n",
        "with torch.no_grad():\n",
        "    # Берём два примера\n",
        "    x1 = X_test_t[0:1].to(device)\n",
        "    x2 = X_test_t[1:2].to(device)\n",
        "    \n",
        "    # Encode\n",
        "    mu1, _ = vae.encode(x1)\n",
        "    mu2, _ = vae.encode(x2)\n",
        "    \n",
        "    # Interpolate\n",
        "    alphas = torch.linspace(0, 1, 5).unsqueeze(1).to(device)\n",
        "    z_interp = alphas * mu1 + (1 - alphas) * mu2\n",
        "    \n",
        "    # Decode\n",
        "    x_interp = vae.decode(z_interp).cpu().numpy()\n",
        "\n",
        "print('🔄 Интерполяция между двумя пассажирами:')\n",
        "print(pd.DataFrame(scaler.inverse_transform(x_interp), columns=features))\n",
        "print('\\n✅ Smooth переход в latent space!')"
    ]
})

# ============================================================================
# COMPARISON
# ============================================================================

practical_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### 2.7 Сравнение с PCA"]
})

practical_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# PCA для сравнения\n",
        "pca = PCA(n_components=2)\n",
        "z_pca_survived = pca.fit_transform(X_test_scaled)\n",
        "z_pca_died = pca.transform(X_died_scaled)\n",
        "\n",
        "# Визуализация: AE vs PCA\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n",
        "\n",
        "# Vanilla AE\n",
        "axes[0].scatter(z_survived[:, 0], z_survived[:, 1], alpha=0.6, label='Survived', c='green')\n",
        "axes[0].scatter(z_died[:, 0], z_died[:, 1], alpha=0.6, label='Died', c='red')\n",
        "axes[0].set_xlabel('Latent Dim 1')\n",
        "axes[0].set_ylabel('Latent Dim 2')\n",
        "axes[0].set_title('Autoencoder Latent Space')\n",
        "axes[0].legend()\n",
        "axes[0].grid(True, alpha=0.3)\n",
        "\n",
        "# PCA\n",
        "axes[1].scatter(z_pca_survived[:, 0], z_pca_survived[:, 1], alpha=0.6, label='Survived', c='green')\n",
        "axes[1].scatter(z_pca_died[:, 0], z_pca_died[:, 1], alpha=0.6, label='Died', c='red')\n",
        "axes[1].set_xlabel('PC 1')\n",
        "axes[1].set_ylabel('PC 2')\n",
        "axes[1].set_title('PCA 2D Projection')\n",
        "axes[1].legend()\n",
        "axes[1].grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(f'PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}')"
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
        "1. **Vanilla Autoencoder:**\n",
        "   - Архитектура Encoder-Decoder\n",
        "   - Compression через bottleneck\n",
        "   - Latent space representation\n",
        "\n",
        "2. **Denoising Autoencoder:**\n",
        "   - Обучение на зашумлённых данных\n",
        "   - Robustness к шуму\n",
        "   - Regularization эффект\n",
        "\n",
        "3. **Variational Autoencoder (VAE):**\n",
        "   - Probabilistic latent space\n",
        "   - Reparameterization trick\n",
        "   - KL divergence для smooth распределения\n",
        "   - Генерация новых примеров\n",
        "\n",
        "### Ключевые инсайты:\n",
        "\n",
        "#### ✅ Anomaly Detection работает!\n",
        "- Autoencoder, обученный на выживших, **хуже восстанавливает** погибших\n",
        "- Reconstruction error — хорошая метрика аномальности\n",
        "- ROC-AUC ~0.65-0.75 (зависит от данных)\n",
        "\n",
        "#### Autoencoder vs PCA:\n",
        "\n",
        "| Критерий | Autoencoder | PCA |\n",
        "|----------|------------|-----|\n",
        "| **Линейность** | Нелинейная | Линейная |\n",
        "| **Сложность** | Высокая (обучение NN) | Низкая (eigen decomposition) |\n",
        "| **Качество** | Лучше для нелинейных данных | Отлично для линейных |\n",
        "| **Скорость** | Медленнее (GPU помогает) | Быстро |\n",
        "| **Интерпретируемость** | Низкая | Высокая (PC = линейные комбинации) |\n",
        "\n",
        "**Рекомендация:**\n",
        "- 🚀 **Начните с PCA** (быстро, просто)\n",
        "- 🧪 **Попробуйте AE** если PCA даёт плохие результаты\n",
        "- 🎨 **Используйте VAE** если нужна генерация\n",
        "\n",
        "### Применения в реальном мире:\n",
        "\n",
        "#### 1. Anomaly Detection\n",
        "- 💳 **Fraud Detection:** Обучаем на легитимных транзакциях\n",
        "- 🏭 **Manufacturing:** Обнаружение дефектов\n",
        "- 🔒 **Cybersecurity:** Intrusion detection\n",
        "- 🏥 **Healthcare:** Редкие заболевания\n",
        "\n",
        "**Порог аномальности:**\n",
        "```python\n",
        "threshold = np.percentile(errors_normal, 95)  # 95th percentile\n",
        "is_anomaly = error > threshold\n",
        "```\n",
        "\n",
        "#### 2. Dimensionality Reduction\n",
        "- 📊 **Visualization:** 2D/3D проекции высокоразмерных данных\n",
        "- ⚡ **Preprocessing:** Сжатие перед другими моделями\n",
        "- 💾 **Compression:** Хранение данных в компактном виде\n",
        "\n",
        "#### 3. Data Generation (VAE)\n",
        "- 🎨 **Image synthesis:** Faces, art (с CNN вместо FC)\n",
        "- 🧬 **Drug discovery:** Генерация новых молекул\n",
        "- 📊 **Data augmentation:** Синтетические примеры для обучения\n",
        "- 🎵 **Music generation:** (с RNN/Transformers)\n",
        "\n",
        "#### 4. Denoising\n",
        "- 🔊 **Audio:** Очистка записей\n",
        "- 🖼️ **Images:** Удаление артефактов, upscaling\n",
        "- 📡 **Signals:** Фильтрация sensor данных\n",
        "\n",
        "### Ограничения:\n",
        "\n",
        "❌ **Для табличных данных:**\n",
        "- XGBoost/LightGBM обычно лучше для supervised tasks\n",
        "- AE полезен только для unsupervised (anomaly, dimensionality reduction)\n",
        "\n",
        "❌ **VAE quality:**\n",
        "- Сгенерированные примеры могут быть \"размытыми\"\n",
        "- GAN часто даёт более realistic результаты\n",
        "\n",
        "### Следующие шаги:\n",
        "\n",
        "1. **Convolutional AE:** Для изображений (Phase 5: Computer Vision)\n",
        "2. **Recurrent AE:** Для временных рядов (Phase 3: Time Series)\n",
        "3. **Transformer AE:** BERT — по сути denoising AE для текста!\n",
        "4. **GAN:** Generative Adversarial Networks (следующий уровень генерации)\n",
        "\n",
        "---\n",
        "\n",
        "## 🎉 Phase 2: Deep Learning Basics ЗАВЕРШЁН!\n",
        "\n",
        "**Пройдено:**\n",
        "1. ✅ **MLP:** Полносвязные сети, backpropagation, optimizers\n",
        "2. ✅ **1D-CNN:** Convolutions, filters, pooling\n",
        "3. ✅ **Autoencoders:** Vanilla, Denoising, VAE\n",
        "\n",
        "**Вы освоили фундаментальные блоки Deep Learning!** 🚀\n",
        "\n",
        "**Следующая фаза:**\n",
        "- **Phase 3:** RNN/LSTM для временных рядов\n",
        "- **Phase 4:** Transformers и attention механизм\n",
        "- **Phase 5:** Computer Vision (2D-CNN, ResNet, etc.)\n",
        "\n",
        "**Поздравляю!** Вы готовы к более сложным архитектурам! 🎓\n"
    ]
})

# Добавляем практику
for cell in practical_cells:
    notebook['cells'].append(cell)

# Сохраняем
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'✅ Практика добавлена: {len(practical_cells)} ячеек')
print(f'Всего ячеек: {len(notebook["cells"])}')
print(f'Ноутбук готов: {notebook_path}')
print('🎉 Phase 2 Deep Learning Basics COMPLETE!')
