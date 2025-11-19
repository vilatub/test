#!/usr/bin/env python3
"""
Скрипт для создания 03_ppo_advanced.ipynb
Proximal Policy Optimization
"""

import json

def create_notebook():
    cells = []

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 9: Reinforcement Learning\n",
            "## Часть 3: Proximal Policy Optimization (PPO)\n",
            "\n",
            "### В этом ноутбуке:\n",
            "\n",
            "1. **PPO** - clipped objective function\n",
            "2. **GAE** - Generalized Advantage Estimation\n",
            "3. **Multiple epochs** - повторное использование данных\n",
            "4. **Практическое применение**"
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
            "from torch.distributions import Categorical\n",
            "import matplotlib.pyplot as plt\n",
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
            "## 1. PPO Motivation\n",
            "\n",
            "### Проблема Policy Gradient\n",
            "\n",
            "- Большие обновления могут разрушить policy\n",
            "- Нужен способ ограничить изменения\n",
            "\n",
            "### PPO Solution\n",
            "\n",
            "Clipped surrogate objective:\n",
            "\n",
            "$$L^{CLIP}(\\theta) = \\mathbb{E}[\\min(r_t(\\theta) A_t, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) A_t)]$$\n",
            "\n",
            "где $r_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{old}}(a_t|s_t)}$"
        ]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Среда"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "class SimpleCartPole:\n",
            "    \"\"\"Упрощённая симуляция CartPole\"\"\"\n",
            "    \n",
            "    def __init__(self):\n",
            "        self.gravity = 9.8\n",
            "        self.masscart = 1.0\n",
            "        self.masspole = 0.1\n",
            "        self.total_mass = self.masspole + self.masscart\n",
            "        self.length = 0.5\n",
            "        self.polemass_length = self.masspole * self.length\n",
            "        self.force_mag = 10.0\n",
            "        self.tau = 0.02\n",
            "        \n",
            "        self.x_threshold = 2.4\n",
            "        self.theta_threshold = 12 * np.pi / 180\n",
            "        \n",
            "        self.state = None\n",
            "        self.steps_count = 0\n",
            "    \n",
            "    def reset(self):\n",
            "        self.state = np.random.uniform(-0.05, 0.05, size=4)\n",
            "        self.steps_count = 0\n",
            "        return self.state.copy()\n",
            "    \n",
            "    def step(self, action):\n",
            "        x, x_dot, theta, theta_dot = self.state\n",
            "        \n",
            "        force = self.force_mag if action == 1 else -self.force_mag\n",
            "        costheta = np.cos(theta)\n",
            "        sintheta = np.sin(theta)\n",
            "        \n",
            "        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass\n",
            "        thetaacc = (self.gravity * sintheta - costheta * temp) / \\\n",
            "                   (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))\n",
            "        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass\n",
            "        \n",
            "        x = x + self.tau * x_dot\n",
            "        x_dot = x_dot + self.tau * xacc\n",
            "        theta = theta + self.tau * theta_dot\n",
            "        theta_dot = theta_dot + self.tau * thetaacc\n",
            "        \n",
            "        self.state = np.array([x, x_dot, theta, theta_dot])\n",
            "        self.steps_count += 1\n",
            "        \n",
            "        done = bool(\n",
            "            x < -self.x_threshold or x > self.x_threshold or\n",
            "            theta < -self.theta_threshold or theta > self.theta_threshold or\n",
            "            self.steps_count >= 500\n",
            "        )\n",
            "        \n",
            "        reward = 1.0 if not done else 0.0\n",
            "        return self.state.copy(), reward, done\n",
            "\n",
            "env = SimpleCartPole()\n",
            "print('Environment ready')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. PPO Network"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "class PPONetwork(nn.Module):\n",
            "    \"\"\"Actor-Critic Network для PPO\"\"\"\n",
            "    \n",
            "    def __init__(self, state_dim, action_dim, hidden_dim=64):\n",
            "        super().__init__()\n",
            "        \n",
            "        # Actor\n",
            "        self.actor = nn.Sequential(\n",
            "            nn.Linear(state_dim, hidden_dim),\n",
            "            nn.Tanh(),\n",
            "            nn.Linear(hidden_dim, hidden_dim),\n",
            "            nn.Tanh(),\n",
            "            nn.Linear(hidden_dim, action_dim),\n",
            "            nn.Softmax(dim=-1)\n",
            "        )\n",
            "        \n",
            "        # Critic\n",
            "        self.critic = nn.Sequential(\n",
            "            nn.Linear(state_dim, hidden_dim),\n",
            "            nn.Tanh(),\n",
            "            nn.Linear(hidden_dim, hidden_dim),\n",
            "            nn.Tanh(),\n",
            "            nn.Linear(hidden_dim, 1)\n",
            "        )\n",
            "    \n",
            "    def forward(self, x):\n",
            "        return self.actor(x), self.critic(x)\n",
            "    \n",
            "    def get_action(self, state):\n",
            "        state = torch.FloatTensor(state).unsqueeze(0).to(device)\n",
            "        probs, value = self.forward(state)\n",
            "        dist = Categorical(probs)\n",
            "        action = dist.sample()\n",
            "        return action.item(), dist.log_prob(action), value\n",
            "    \n",
            "    def evaluate(self, states, actions):\n",
            "        probs, values = self.forward(states)\n",
            "        dist = Categorical(probs)\n",
            "        log_probs = dist.log_prob(actions)\n",
            "        entropy = dist.entropy()\n",
            "        return log_probs, values.squeeze(), entropy\n",
            "\n",
            "print('PPONetwork создан')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. PPO Agent"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "class PPOAgent:\n",
            "    \"\"\"Proximal Policy Optimization Agent\"\"\"\n",
            "    \n",
            "    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, \n",
            "                 gae_lambda=0.95, clip_epsilon=0.2, \n",
            "                 value_coef=0.5, entropy_coef=0.01,\n",
            "                 n_epochs=4, batch_size=64):\n",
            "        \n",
            "        self.network = PPONetwork(state_dim, action_dim).to(device)\n",
            "        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)\n",
            "        \n",
            "        self.gamma = gamma\n",
            "        self.gae_lambda = gae_lambda\n",
            "        self.clip_epsilon = clip_epsilon\n",
            "        self.value_coef = value_coef\n",
            "        self.entropy_coef = entropy_coef\n",
            "        self.n_epochs = n_epochs\n",
            "        self.batch_size = batch_size\n",
            "        \n",
            "        # Buffer\n",
            "        self.states = []\n",
            "        self.actions = []\n",
            "        self.log_probs = []\n",
            "        self.rewards = []\n",
            "        self.values = []\n",
            "        self.dones = []\n",
            "    \n",
            "    def select_action(self, state):\n",
            "        action, log_prob, value = self.network.get_action(state)\n",
            "        \n",
            "        self.states.append(state)\n",
            "        self.actions.append(action)\n",
            "        self.log_probs.append(log_prob.item())\n",
            "        self.values.append(value.item())\n",
            "        \n",
            "        return action\n",
            "    \n",
            "    def store_outcome(self, reward, done):\n",
            "        self.rewards.append(reward)\n",
            "        self.dones.append(done)\n",
            "    \n",
            "    def compute_gae(self, next_value):\n",
            "        \"\"\"Generalized Advantage Estimation\"\"\"\n",
            "        \n",
            "        advantages = []\n",
            "        gae = 0\n",
            "        \n",
            "        values = self.values + [next_value]\n",
            "        \n",
            "        for t in reversed(range(len(self.rewards))):\n",
            "            if self.dones[t]:\n",
            "                delta = self.rewards[t] - values[t]\n",
            "                gae = delta\n",
            "            else:\n",
            "                delta = self.rewards[t] + self.gamma * values[t+1] - values[t]\n",
            "                gae = delta + self.gamma * self.gae_lambda * gae\n",
            "            \n",
            "            advantages.insert(0, gae)\n",
            "        \n",
            "        return advantages\n",
            "    \n",
            "    def update(self):\n",
            "        \"\"\"PPO Update\"\"\"\n",
            "        \n",
            "        # Compute advantages\n",
            "        with torch.no_grad():\n",
            "            last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(device)\n",
            "            _, next_value = self.network(last_state)\n",
            "            next_value = next_value.item() if not self.dones[-1] else 0\n",
            "        \n",
            "        advantages = self.compute_gae(next_value)\n",
            "        returns = [adv + val for adv, val in zip(advantages, self.values)]\n",
            "        \n",
            "        # Convert to tensors\n",
            "        states = torch.FloatTensor(np.array(self.states)).to(device)\n",
            "        actions = torch.LongTensor(self.actions).to(device)\n",
            "        old_log_probs = torch.FloatTensor(self.log_probs).to(device)\n",
            "        advantages = torch.FloatTensor(advantages).to(device)\n",
            "        returns = torch.FloatTensor(returns).to(device)\n",
            "        \n",
            "        # Normalize advantages\n",
            "        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)\n",
            "        \n",
            "        # PPO epochs\n",
            "        total_loss = 0\n",
            "        for _ in range(self.n_epochs):\n",
            "            # Mini-batch updates\n",
            "            indices = np.random.permutation(len(states))\n",
            "            \n",
            "            for start in range(0, len(states), self.batch_size):\n",
            "                end = start + self.batch_size\n",
            "                batch_indices = indices[start:end]\n",
            "                \n",
            "                batch_states = states[batch_indices]\n",
            "                batch_actions = actions[batch_indices]\n",
            "                batch_old_log_probs = old_log_probs[batch_indices]\n",
            "                batch_advantages = advantages[batch_indices]\n",
            "                batch_returns = returns[batch_indices]\n",
            "                \n",
            "                # Evaluate current policy\n",
            "                log_probs, values, entropy = self.network.evaluate(\n",
            "                    batch_states, batch_actions\n",
            "                )\n",
            "                \n",
            "                # Ratio\n",
            "                ratio = torch.exp(log_probs - batch_old_log_probs)\n",
            "                \n",
            "                # Clipped surrogate loss\n",
            "                surr1 = ratio * batch_advantages\n",
            "                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, \n",
            "                                   1 + self.clip_epsilon) * batch_advantages\n",
            "                actor_loss = -torch.min(surr1, surr2).mean()\n",
            "                \n",
            "                # Value loss\n",
            "                value_loss = F.mse_loss(values, batch_returns)\n",
            "                \n",
            "                # Entropy bonus\n",
            "                entropy_loss = -entropy.mean()\n",
            "                \n",
            "                # Total loss\n",
            "                loss = (actor_loss + \n",
            "                       self.value_coef * value_loss + \n",
            "                       self.entropy_coef * entropy_loss)\n",
            "                \n",
            "                self.optimizer.zero_grad()\n",
            "                loss.backward()\n",
            "                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)\n",
            "                self.optimizer.step()\n",
            "                \n",
            "                total_loss += loss.item()\n",
            "        \n",
            "        # Clear buffer\n",
            "        self.states = []\n",
            "        self.actions = []\n",
            "        self.log_probs = []\n",
            "        self.rewards = []\n",
            "        self.values = []\n",
            "        self.dones = []\n",
            "        \n",
            "        return total_loss\n",
            "\n",
            "print('PPOAgent создан')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Обучение PPO"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def train_ppo(env, agent, episodes=500, update_freq=2048):\n",
            "    \"\"\"Обучение PPO агента\"\"\"\n",
            "    \n",
            "    rewards_history = []\n",
            "    episode_reward = 0\n",
            "    episode_count = 0\n",
            "    steps = 0\n",
            "    \n",
            "    state = env.reset()\n",
            "    \n",
            "    while episode_count < episodes:\n",
            "        action = agent.select_action(state)\n",
            "        next_state, reward, done = env.step(action)\n",
            "        agent.store_outcome(reward, done)\n",
            "        \n",
            "        episode_reward += reward\n",
            "        steps += 1\n",
            "        \n",
            "        if done:\n",
            "            rewards_history.append(episode_reward)\n",
            "            episode_reward = 0\n",
            "            episode_count += 1\n",
            "            state = env.reset()\n",
            "            \n",
            "            if episode_count % 100 == 0:\n",
            "                avg_reward = np.mean(rewards_history[-100:])\n",
            "                print(f'Episode {episode_count}, Avg Reward: {avg_reward:.1f}')\n",
            "        else:\n",
            "            state = next_state\n",
            "        \n",
            "        # Update\n",
            "        if steps % update_freq == 0 and len(agent.states) > 0:\n",
            "            agent.update()\n",
            "    \n",
            "    # Final update\n",
            "    if len(agent.states) > 0:\n",
            "        agent.update()\n",
            "    \n",
            "    return rewards_history\n",
            "\n",
            "# Обучение\n",
            "env = SimpleCartPole()\n",
            "ppo_agent = PPOAgent(\n",
            "    state_dim=4,\n",
            "    action_dim=2,\n",
            "    lr=3e-4,\n",
            "    gamma=0.99,\n",
            "    gae_lambda=0.95,\n",
            "    clip_epsilon=0.2,\n",
            "    n_epochs=4,\n",
            "    batch_size=64\n",
            ")\n",
            "\n",
            "print('Обучение PPO...\\n')\n",
            "ppo_rewards = train_ppo(env, ppo_agent, episodes=500, update_freq=512)"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Визуализация\n",
            "plt.figure(figsize=(12, 4))\n",
            "\n",
            "plt.plot(ppo_rewards, alpha=0.3, label='Episode Reward')\n",
            "if len(ppo_rewards) >= 50:\n",
            "    smooth = np.convolve(ppo_rewards, np.ones(50)/50, mode='valid')\n",
            "    plt.plot(range(49, len(ppo_rewards)), smooth, linewidth=2, label='Moving Avg (50)')\n",
            "\n",
            "plt.xlabel('Episode')\n",
            "plt.ylabel('Total Reward')\n",
            "plt.title('PPO Training on CartPole')\n",
            "plt.legend()\n",
            "plt.show()\n",
            "\n",
            "print(f'\\nФинальная средняя награда: {np.mean(ppo_rewards[-100:]):.1f}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Анализ компонентов PPO"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Визуализация clipping\n",
            "epsilon = 0.2\n",
            "ratios = np.linspace(0.5, 1.5, 100)\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "\n",
            "# Positive advantage\n",
            "advantage = 1.0\n",
            "surr1 = ratios * advantage\n",
            "surr2 = np.clip(ratios, 1 - epsilon, 1 + epsilon) * advantage\n",
            "objective = np.minimum(surr1, surr2)\n",
            "\n",
            "axes[0].plot(ratios, surr1, '--', label='Unclipped', alpha=0.7)\n",
            "axes[0].plot(ratios, surr2, '--', label='Clipped', alpha=0.7)\n",
            "axes[0].plot(ratios, objective, linewidth=2, label='PPO Objective')\n",
            "axes[0].axvline(x=1, color='gray', linestyle=':')\n",
            "axes[0].set_xlabel('Probability Ratio')\n",
            "axes[0].set_ylabel('Objective')\n",
            "axes[0].set_title('Positive Advantage (A > 0)')\n",
            "axes[0].legend()\n",
            "\n",
            "# Negative advantage\n",
            "advantage = -1.0\n",
            "surr1 = ratios * advantage\n",
            "surr2 = np.clip(ratios, 1 - epsilon, 1 + epsilon) * advantage\n",
            "objective = np.minimum(surr1, surr2)\n",
            "\n",
            "axes[1].plot(ratios, surr1, '--', label='Unclipped', alpha=0.7)\n",
            "axes[1].plot(ratios, surr2, '--', label='Clipped', alpha=0.7)\n",
            "axes[1].plot(ratios, objective, linewidth=2, label='PPO Objective')\n",
            "axes[1].axvline(x=1, color='gray', linestyle=':')\n",
            "axes[1].set_xlabel('Probability Ratio')\n",
            "axes[1].set_ylabel('Objective')\n",
            "axes[1].set_title('Negative Advantage (A < 0)')\n",
            "axes[1].legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print('PPO clipping предотвращает слишком большие изменения policy')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Практические советы\n",
            "\n",
            "### Гиперпараметры PPO:\n",
            "\n",
            "| Параметр | Типичное значение | Описание |\n",
            "|----------|-------------------|----------|\n",
            "| clip_epsilon | 0.1-0.3 | Ограничение изменения policy |\n",
            "| gamma | 0.99 | Discount factor |\n",
            "| gae_lambda | 0.95 | GAE parameter |\n",
            "| n_epochs | 3-10 | Эпохи на батч данных |\n",
            "| batch_size | 32-512 | Размер мини-батча |\n",
            "\n",
            "### Best Practices:\n",
            "\n",
            "1. **Нормализация** - observations и advantages\n",
            "2. **Gradient clipping** - стабильность\n",
            "3. **Learning rate schedule** - decay\n",
            "4. **Entropy bonus** - exploration"
        ]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Итоги Phase 9\n",
            "\n",
            "### Изученные алгоритмы:\n",
            "\n",
            "1. **DQN** - Q-learning с нейросетью\n",
            "2. **REINFORCE** - Monte Carlo policy gradient\n",
            "3. **Actor-Critic** - комбинация value и policy\n",
            "4. **PPO** - современный стандарт RL\n",
            "\n",
            "### Сравнение:\n",
            "\n",
            "| Метод | Sample Eff. | Stability | Continuous |\n",
            "|-------|-------------|-----------|------------|\n",
            "| DQN | High | Medium | No |\n",
            "| REINFORCE | Low | Low | Yes |\n",
            "| A2C | Medium | Medium | Yes |\n",
            "| PPO | Medium | High | Yes |\n",
            "\n",
            "### Применение в ML:\n",
            "\n",
            "- **Robotics** - управление роботами\n",
            "- **Games** - Atari, Go, StarCraft\n",
            "- **NLP** - RLHF для LLM\n",
            "- **Trading** - оптимизация портфеля\n",
            "\n",
            "### Дальнейшее изучение:\n",
            "\n",
            "- **SAC** - Soft Actor-Critic\n",
            "- **TD3** - Twin Delayed DDPG\n",
            "- **Model-based RL** - Dreamer, MuZero"
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
    output_path = "/home/user/test/notebooks/phase9_reinforcement_learning/03_ppo_advanced.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"Notebook created: {output_path}")
    print(f"Total cells: {len(notebook['cells'])}")
