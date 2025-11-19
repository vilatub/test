#!/usr/bin/env python3
"""
Скрипт для создания 03_diffusion_models.ipynb
Diffusion Models (DDPM)
"""

import json

def create_notebook():
    cells = []

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 10: Generative Models\n",
            "## Часть 3: Diffusion Models\n",
            "\n",
            "### В этом ноутбуке:\n",
            "\n",
            "1. **Diffusion process** - постепенное добавление шума\n",
            "2. **DDPM** - Denoising Diffusion Probabilistic Models\n",
            "3. **Forward и Reverse process**\n",
            "4. **Noise prediction** - обучение модели\n",
            "5. **Sampling** - генерация изображений"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import numpy as np\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.optim as optim\n",
            "import torch.nn.functional as F\n",
            "import matplotlib.pyplot as plt\n",
            "from torch.utils.data import DataLoader, TensorDataset\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "torch.manual_seed(42)\n",
            "np.random.seed(42)\n",
            "\n",
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
            "## 1. Diffusion Process\n",
            "\n",
            "### Forward Process (добавление шума)\n",
            "\n",
            "Постепенно превращаем данные в шум:\n",
            "\n",
            "$$q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1-\\beta_t} x_{t-1}, \\beta_t I)$$\n",
            "\n",
            "### Reverse Process (удаление шума)\n",
            "\n",
            "Модель учится предсказывать шум:\n",
            "\n",
            "$$p_\\theta(x_{t-1} | x_t) = \\mathcal{N}(x_{t-1}; \\mu_\\theta(x_t, t), \\sigma_t^2 I)$$"
        ]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Данные"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def generate_patterns(n_samples=2000):\n",
            "    \"\"\"Генерация простых паттернов 8x8\"\"\"\n",
            "    patterns = []\n",
            "    \n",
            "    for _ in range(n_samples):\n",
            "        img = np.zeros((8, 8))\n",
            "        pattern_type = np.random.randint(4)\n",
            "        \n",
            "        if pattern_type == 0:\n",
            "            row = np.random.randint(1, 7)\n",
            "            img[row, 1:7] = 1\n",
            "        elif pattern_type == 1:\n",
            "            col = np.random.randint(1, 7)\n",
            "            img[1:7, col] = 1\n",
            "        elif pattern_type == 2:\n",
            "            size = np.random.randint(2, 4)\n",
            "            start = np.random.randint(1, 6-size)\n",
            "            img[start:start+size, start:start+size] = 1\n",
            "        else:\n",
            "            for i in range(6):\n",
            "                img[i+1, i+1] = 1\n",
            "        \n",
            "        patterns.append(img)\n",
            "    \n",
            "    return np.array(patterns)\n",
            "\n",
            "# Данные\n",
            "X = generate_patterns(2000)\n",
            "X = torch.FloatTensor(X).view(-1, 1, 8, 8)\n",
            "# Нормализация к [-1, 1]\n",
            "X = X * 2 - 1\n",
            "\n",
            "dataset = TensorDataset(X)\n",
            "dataloader = DataLoader(dataset, batch_size=64, shuffle=True)\n",
            "\n",
            "print(f'Data shape: {X.shape}')\n",
            "print(f'Value range: [{X.min():.1f}, {X.max():.1f}]')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Noise Schedule"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "class DiffusionSchedule:\n",
            "    \"\"\"Linear beta schedule для DDPM\"\"\"\n",
            "    \n",
            "    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02):\n",
            "        self.timesteps = timesteps\n",
            "        \n",
            "        # Linear schedule\n",
            "        self.betas = torch.linspace(beta_start, beta_end, timesteps)\n",
            "        \n",
            "        # Pre-compute useful quantities\n",
            "        self.alphas = 1 - self.betas\n",
            "        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)\n",
            "        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)\n",
            "        \n",
            "        # For q(x_t | x_0)\n",
            "        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)\n",
            "        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)\n",
            "        \n",
            "        # For posterior q(x_{t-1} | x_t, x_0)\n",
            "        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)\n",
            "    \n",
            "    def get_index(self, vals, t, x_shape):\n",
            "        \"\"\"Extract value at timestep t\"\"\"\n",
            "        batch_size = t.shape[0]\n",
            "        out = vals.gather(-1, t.cpu())\n",
            "        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)\n",
            "\n",
            "# Создаём schedule\n",
            "schedule = DiffusionSchedule(timesteps=100)\n",
            "\n",
            "# Визуализация\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "\n",
            "axes[0].plot(schedule.betas.numpy())\n",
            "axes[0].set_xlabel('Timestep')\n",
            "axes[0].set_ylabel('Beta')\n",
            "axes[0].set_title('Noise Schedule (beta)')\n",
            "\n",
            "axes[1].plot(schedule.alphas_cumprod.numpy())\n",
            "axes[1].set_xlabel('Timestep')\n",
            "axes[1].set_ylabel('Alpha cumulative product')\n",
            "axes[1].set_title('Signal Remaining')\n",
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
            "## 4. Forward Diffusion"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def forward_diffusion(x_0, t, schedule, noise=None):\n",
            "    \"\"\"\n",
            "    Forward diffusion: q(x_t | x_0)\n",
            "    \n",
            "    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise\n",
            "    \"\"\"\n",
            "    if noise is None:\n",
            "        noise = torch.randn_like(x_0)\n",
            "    \n",
            "    sqrt_alphas_cumprod_t = schedule.get_index(\n",
            "        schedule.sqrt_alphas_cumprod, t, x_0.shape\n",
            "    )\n",
            "    sqrt_one_minus_alphas_cumprod_t = schedule.get_index(\n",
            "        schedule.sqrt_one_minus_alphas_cumprod, t, x_0.shape\n",
            "    )\n",
            "    \n",
            "    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise\n",
            "\n",
            "# Визуализация forward diffusion\n",
            "sample_img = X[0:1]\n",
            "timesteps_to_show = [0, 10, 25, 50, 75, 99]\n",
            "\n",
            "fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(12, 2))\n",
            "\n",
            "for i, t in enumerate(timesteps_to_show):\n",
            "    t_tensor = torch.tensor([t])\n",
            "    noisy, _ = forward_diffusion(sample_img, t_tensor, schedule)\n",
            "    \n",
            "    # Denormalize для визуализации\n",
            "    img = (noisy[0, 0] + 1) / 2\n",
            "    axes[i].imshow(img.numpy(), cmap='gray', vmin=0, vmax=1)\n",
            "    axes[i].set_title(f't={t}')\n",
            "    axes[i].axis('off')\n",
            "\n",
            "plt.suptitle('Forward Diffusion Process')\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Noise Prediction Network"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "class SinusoidalPositionEmbeddings(nn.Module):\n",
            "    \"\"\"Позиционные эмбеддинги для timestep\"\"\"\n",
            "    \n",
            "    def __init__(self, dim):\n",
            "        super().__init__()\n",
            "        self.dim = dim\n",
            "    \n",
            "    def forward(self, time):\n",
            "        device = time.device\n",
            "        half_dim = self.dim // 2\n",
            "        embeddings = np.log(10000) / (half_dim - 1)\n",
            "        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)\n",
            "        embeddings = time[:, None] * embeddings[None, :]\n",
            "        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)\n",
            "        return embeddings\n",
            "\n",
            "class SimpleUNet(nn.Module):\n",
            "    \"\"\"Упрощённая U-Net для предсказания шума\"\"\"\n",
            "    \n",
            "    def __init__(self, in_channels=1, time_emb_dim=32):\n",
            "        super().__init__()\n",
            "        \n",
            "        # Time embedding\n",
            "        self.time_mlp = nn.Sequential(\n",
            "            SinusoidalPositionEmbeddings(time_emb_dim),\n",
            "            nn.Linear(time_emb_dim, time_emb_dim),\n",
            "            nn.ReLU()\n",
            "        )\n",
            "        \n",
            "        # Encoder\n",
            "        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)\n",
            "        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)\n",
            "        \n",
            "        # Middle (with time conditioning)\n",
            "        self.time_proj = nn.Linear(time_emb_dim, 64)\n",
            "        self.conv_mid = nn.Conv2d(64, 64, 3, padding=1)\n",
            "        \n",
            "        # Decoder\n",
            "        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)\n",
            "        self.conv4 = nn.Conv2d(32, in_channels, 3, padding=1)\n",
            "        \n",
            "        self.relu = nn.ReLU()\n",
            "    \n",
            "    def forward(self, x, t):\n",
            "        # Time embedding\n",
            "        t_emb = self.time_mlp(t)\n",
            "        \n",
            "        # Encoder\n",
            "        h1 = self.relu(self.conv1(x))\n",
            "        h2 = self.relu(self.conv2(h1))\n",
            "        \n",
            "        # Add time conditioning\n",
            "        t_proj = self.time_proj(t_emb)[:, :, None, None]\n",
            "        h2 = h2 + t_proj\n",
            "        \n",
            "        # Middle\n",
            "        h_mid = self.relu(self.conv_mid(h2))\n",
            "        \n",
            "        # Decoder\n",
            "        h3 = self.relu(self.conv3(h_mid))\n",
            "        out = self.conv4(h3)\n",
            "        \n",
            "        return out\n",
            "\n",
            "# Тест\n",
            "model = SimpleUNet()\n",
            "test_x = torch.randn(2, 1, 8, 8)\n",
            "test_t = torch.tensor([10, 50])\n",
            "out = model(test_x, test_t)\n",
            "print(f'Input: {test_x.shape}, Output: {out.shape}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Training Loop"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def train_diffusion(model, dataloader, schedule, epochs=100, lr=1e-3):\n",
            "    \"\"\"Обучение Diffusion Model\"\"\"\n",
            "    \n",
            "    model = model.to(device)\n",
            "    optimizer = optim.Adam(model.parameters(), lr=lr)\n",
            "    \n",
            "    losses = []\n",
            "    \n",
            "    for epoch in range(epochs):\n",
            "        model.train()\n",
            "        epoch_loss = 0\n",
            "        \n",
            "        for batch in dataloader:\n",
            "            x_0 = batch[0].to(device)\n",
            "            batch_size = x_0.shape[0]\n",
            "            \n",
            "            # Random timesteps\n",
            "            t = torch.randint(0, schedule.timesteps, (batch_size,), device=device)\n",
            "            \n",
            "            # Add noise\n",
            "            noise = torch.randn_like(x_0)\n",
            "            x_t, _ = forward_diffusion(x_0, t, schedule, noise)\n",
            "            \n",
            "            # Predict noise\n",
            "            noise_pred = model(x_t, t)\n",
            "            \n",
            "            # Loss\n",
            "            loss = F.mse_loss(noise_pred, noise)\n",
            "            \n",
            "            optimizer.zero_grad()\n",
            "            loss.backward()\n",
            "            optimizer.step()\n",
            "            \n",
            "            epoch_loss += loss.item()\n",
            "        \n",
            "        avg_loss = epoch_loss / len(dataloader)\n",
            "        losses.append(avg_loss)\n",
            "        \n",
            "        if (epoch + 1) % 20 == 0:\n",
            "            print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}')\n",
            "    \n",
            "    return losses\n",
            "\n",
            "# Обучение\n",
            "model = SimpleUNet()\n",
            "schedule = DiffusionSchedule(timesteps=100)\n",
            "\n",
            "print('Обучение Diffusion Model...\\n')\n",
            "losses = train_diffusion(model, dataloader, schedule, epochs=100, lr=1e-3)"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Визуализация обучения\n",
            "plt.figure(figsize=(10, 4))\n",
            "plt.plot(losses)\n",
            "plt.xlabel('Epoch')\n",
            "plt.ylabel('MSE Loss')\n",
            "plt.title('Diffusion Model Training')\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Sampling (Reverse Process)"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "@torch.no_grad()\n",
            "def sample(model, schedule, n_samples=16, img_size=(1, 8, 8)):\n",
            "    \"\"\"Генерация через reverse diffusion\"\"\"\n",
            "    \n",
            "    model.eval()\n",
            "    \n",
            "    # Начинаем с чистого шума\n",
            "    x = torch.randn(n_samples, *img_size).to(device)\n",
            "    \n",
            "    # Reverse diffusion\n",
            "    for t in reversed(range(schedule.timesteps)):\n",
            "        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)\n",
            "        \n",
            "        # Predict noise\n",
            "        noise_pred = model(x, t_batch)\n",
            "        \n",
            "        # Get schedule parameters\n",
            "        alpha = schedule.alphas[t]\n",
            "        alpha_cumprod = schedule.alphas_cumprod[t]\n",
            "        beta = schedule.betas[t]\n",
            "        \n",
            "        if t > 0:\n",
            "            noise = torch.randn_like(x)\n",
            "        else:\n",
            "            noise = 0\n",
            "        \n",
            "        # Compute x_{t-1}\n",
            "        x = (1 / torch.sqrt(alpha)) * (\n",
            "            x - (beta / torch.sqrt(1 - alpha_cumprod)) * noise_pred\n",
            "        ) + torch.sqrt(beta) * noise\n",
            "    \n",
            "    # Denormalize\n",
            "    x = (x + 1) / 2\n",
            "    x = torch.clamp(x, 0, 1)\n",
            "    \n",
            "    return x.cpu()\n",
            "\n",
            "# Генерация\n",
            "generated = sample(model, schedule, n_samples=16)\n",
            "\n",
            "# Визуализация\n",
            "fig, axes = plt.subplots(2, 8, figsize=(12, 3))\n",
            "for i in range(16):\n",
            "    ax = axes[i//8, i%8]\n",
            "    ax.imshow(generated[i, 0], cmap='gray')\n",
            "    ax.axis('off')\n",
            "\n",
            "plt.suptitle('Diffusion Model Generated Samples')\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Визуализация Reverse Process"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "@torch.no_grad()\n",
            "def sample_with_trajectory(model, schedule, img_size=(1, 8, 8)):\n",
            "    \"\"\"Генерация с сохранением промежуточных шагов\"\"\"\n",
            "    \n",
            "    model.eval()\n",
            "    x = torch.randn(1, *img_size).to(device)\n",
            "    \n",
            "    trajectory = [((x[0, 0].cpu() + 1) / 2).numpy()]\n",
            "    save_steps = [99, 75, 50, 25, 10, 5, 2, 0]\n",
            "    \n",
            "    for t in reversed(range(schedule.timesteps)):\n",
            "        t_batch = torch.full((1,), t, device=device, dtype=torch.long)\n",
            "        noise_pred = model(x, t_batch)\n",
            "        \n",
            "        alpha = schedule.alphas[t]\n",
            "        alpha_cumprod = schedule.alphas_cumprod[t]\n",
            "        beta = schedule.betas[t]\n",
            "        \n",
            "        noise = torch.randn_like(x) if t > 0 else 0\n",
            "        \n",
            "        x = (1 / torch.sqrt(alpha)) * (\n",
            "            x - (beta / torch.sqrt(1 - alpha_cumprod)) * noise_pred\n",
            "        ) + torch.sqrt(beta) * noise\n",
            "        \n",
            "        if t in save_steps:\n",
            "            img = torch.clamp((x[0, 0].cpu() + 1) / 2, 0, 1)\n",
            "            trajectory.append(img.numpy())\n",
            "    \n",
            "    return trajectory\n",
            "\n",
            "# Визуализация\n",
            "trajectory = sample_with_trajectory(model, schedule)\n",
            "\n",
            "fig, axes = plt.subplots(1, len(trajectory), figsize=(14, 2))\n",
            "timesteps_shown = ['t=99', 't=75', 't=50', 't=25', 't=10', 't=5', 't=2', 't=0']\n",
            "\n",
            "for i, (img, label) in enumerate(zip(trajectory, ['noise'] + timesteps_shown)):\n",
            "    axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)\n",
            "    axes[i].set_title(label)\n",
            "    axes[i].axis('off')\n",
            "\n",
            "plt.suptitle('Reverse Diffusion (Denoising)')\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Итоги Phase 10\n",
            "\n",
            "### Изученные модели:\n",
            "\n",
            "| Модель | Идея | Плюсы | Минусы |\n",
            "|--------|------|-------|--------|\n",
            "| **VAE** | Latent space + reconstruction | Стабильное обучение | Размытые изображения |\n",
            "| **GAN** | Adversarial game | Чёткие изображения | Нестабильность |\n",
            "| **Diffusion** | Постепенное denoising | Высокое качество | Медленный sampling |\n",
            "\n",
            "### Ключевые формулы:\n",
            "\n",
            "**VAE:**\n",
            "$$\\mathcal{L} = \\text{Reconstruction} + D_{KL}$$\n",
            "\n",
            "**GAN:**\n",
            "$$\\min_G \\max_D \\mathbb{E}[\\log D(x)] + \\mathbb{E}[\\log(1 - D(G(z)))]$$\n",
            "\n",
            "**Diffusion:**\n",
            "$$x_{t-1} = \\frac{1}{\\sqrt{\\alpha_t}}\\left(x_t - \\frac{\\beta_t}{\\sqrt{1-\\bar\\alpha_t}}\\epsilon_\\theta(x_t, t)\\right) + \\sigma_t z$$\n",
            "\n",
            "### Современные модели:\n",
            "\n",
            "- **Stable Diffusion** - latent diffusion\n",
            "- **DALL-E** - text-to-image\n",
            "- **Midjourney** - artistic generation\n",
            "\n",
            "### Применение:\n",
            "\n",
            "- Image generation\n",
            "- Image editing (inpainting)\n",
            "- Super-resolution\n",
            "- Text-to-image\n",
            "- Video generation"
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
    output_path = "/home/user/test/notebooks/phase10_generative_models/03_diffusion_models.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"Notebook created: {output_path}")
    print(f"Total cells: {len(notebook['cells'])}")
