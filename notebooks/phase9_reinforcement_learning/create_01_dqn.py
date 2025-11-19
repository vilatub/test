#!/usr/bin/env python3
"""
Скрипт для создания 01_dqn_basics.ipynb
Deep Q-Network основы
"""

import json

def create_notebook():
    cells = []

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 9: Reinforcement Learning\n",
            "## Часть 1: Deep Q-Network (DQN)\n",
            "\n",
            "### В этом ноутбуке:\n",
            "\n",
            "1. **Основы RL** - MDP, Policy, Value Function\n",
            "2. **Q-Learning** - табличный метод\n",
            "3. **DQN** - нейросеть для Q-функции\n",
            "4. **Experience Replay** и **Target Network**\n",
            "5. **Практика** - решение CartPole"
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
            "import matplotlib.pyplot as plt\n",
            "from collections import deque, namedtuple\n",
            "import random\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Для воспроизводимости\n",
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
            "## 1. Основы Reinforcement Learning\n",
            "\n",
            "### Markov Decision Process (MDP)\n",
            "\n",
            "- **State (s)**: состояние среды\n",
            "- **Action (a)**: действие агента\n",
            "- **Reward (r)**: награда за действие\n",
            "- **Policy (π)**: стратегия выбора действий\n",
            "- **Value Function (V)**: ожидаемая сумма наград\n",
            "\n",
            "### Q-Function\n",
            "\n",
            "$$Q(s, a) = \\mathbb{E}[R_t + \\gamma R_{t+1} + \\gamma^2 R_{t+2} + ... | s_t=s, a_t=a]$$\n",
            "\n",
            "Оценивает \"качество\" действия a в состоянии s."
        ]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Простая среда: GridWorld"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "class SimpleGridWorld:\n",
            "    \"\"\"Простой GridWorld 4x4 для демонстрации Q-learning\"\"\"\n",
            "    \n",
            "    def __init__(self):\n",
            "        self.grid_size = 4\n",
            "        self.state = None\n",
            "        self.goal = (3, 3)  # Цель в правом нижнем углу\n",
            "        self.actions = ['up', 'down', 'left', 'right']\n",
            "        self.n_actions = len(self.actions)\n",
            "        self.n_states = self.grid_size ** 2\n",
            "        \n",
            "    def reset(self):\n",
            "        self.state = (0, 0)  # Старт в левом верхнем углу\n",
            "        return self._state_to_idx(self.state)\n",
            "    \n",
            "    def _state_to_idx(self, state):\n",
            "        return state[0] * self.grid_size + state[1]\n",
            "    \n",
            "    def step(self, action_idx):\n",
            "        action = self.actions[action_idx]\n",
            "        x, y = self.state\n",
            "        \n",
            "        # Применяем действие\n",
            "        if action == 'up' and x > 0:\n",
            "            x -= 1\n",
            "        elif action == 'down' and x < self.grid_size - 1:\n",
            "            x += 1\n",
            "        elif action == 'left' and y > 0:\n",
            "            y -= 1\n",
            "        elif action == 'right' and y < self.grid_size - 1:\n",
            "            y += 1\n",
            "        \n",
            "        self.state = (x, y)\n",
            "        \n",
            "        # Награда\n",
            "        if self.state == self.goal:\n",
            "            return self._state_to_idx(self.state), 10.0, True\n",
            "        else:\n",
            "            return self._state_to_idx(self.state), -0.1, False\n",
            "\n",
            "env = SimpleGridWorld()\n",
            "print(f'States: {env.n_states}, Actions: {env.n_actions}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Табличный Q-Learning"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):\n",
            "    \"\"\"Табличный Q-learning\"\"\"\n",
            "    \n",
            "    # Инициализация Q-таблицы\n",
            "    Q = np.zeros((env.n_states, env.n_actions))\n",
            "    rewards_history = []\n",
            "    \n",
            "    for episode in range(episodes):\n",
            "        state = env.reset()\n",
            "        total_reward = 0\n",
            "        done = False\n",
            "        \n",
            "        while not done:\n",
            "            # Epsilon-greedy выбор действия\n",
            "            if np.random.random() < epsilon:\n",
            "                action = np.random.randint(env.n_actions)\n",
            "            else:\n",
            "                action = np.argmax(Q[state])\n",
            "            \n",
            "            # Шаг в среде\n",
            "            next_state, reward, done = env.step(action)\n",
            "            total_reward += reward\n",
            "            \n",
            "            # Q-learning update\n",
            "            best_next = np.max(Q[next_state])\n",
            "            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])\n",
            "            \n",
            "            state = next_state\n",
            "        \n",
            "        rewards_history.append(total_reward)\n",
            "    \n",
            "    return Q, rewards_history\n",
            "\n",
            "# Обучение\n",
            "Q_table, rewards = q_learning(env)\n",
            "\n",
            "# Визуализация обучения\n",
            "plt.figure(figsize=(10, 4))\n",
            "plt.plot(rewards, alpha=0.3)\n",
            "plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), linewidth=2)\n",
            "plt.xlabel('Episode')\n",
            "plt.ylabel('Total Reward')\n",
            "plt.title('Q-Learning Training Progress')\n",
            "plt.show()\n",
            "\n",
            "print(f'\\nСредняя награда (последние 100): {np.mean(rewards[-100:]):.2f}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Визуализация Q-таблицы\n",
            "fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n",
            "\n",
            "for i, action in enumerate(env.actions):\n",
            "    q_values = Q_table[:, i].reshape(env.grid_size, env.grid_size)\n",
            "    im = axes[i].imshow(q_values, cmap='viridis')\n",
            "    axes[i].set_title(f'Q-values for \"{action}\"')\n",
            "    axes[i].set_xticks(range(4))\n",
            "    axes[i].set_yticks(range(4))\n",
            "    plt.colorbar(im, ax=axes[i])\n",
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
            "## 4. Deep Q-Network (DQN)\n",
            "\n",
            "Для сложных сред с большим пространством состояний используем нейросеть.\n",
            "\n",
            "### Ключевые компоненты DQN:\n",
            "\n",
            "1. **Experience Replay** - хранение и переиспользование опыта\n",
            "2. **Target Network** - стабилизация обучения\n",
            "3. **Epsilon-greedy** - баланс exploration/exploitation"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Transition для Experience Replay\n",
            "Transition = namedtuple('Transition', \n",
            "                        ('state', 'action', 'next_state', 'reward', 'done'))\n",
            "\n",
            "class ReplayBuffer:\n",
            "    \"\"\"Experience Replay Buffer\"\"\"\n",
            "    \n",
            "    def __init__(self, capacity):\n",
            "        self.buffer = deque(maxlen=capacity)\n",
            "    \n",
            "    def push(self, *args):\n",
            "        self.buffer.append(Transition(*args))\n",
            "    \n",
            "    def sample(self, batch_size):\n",
            "        transitions = random.sample(self.buffer, batch_size)\n",
            "        return Transition(*zip(*transitions))\n",
            "    \n",
            "    def __len__(self):\n",
            "        return len(self.buffer)\n",
            "\n",
            "print('ReplayBuffer создан')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "class DQN(nn.Module):\n",
            "    \"\"\"Deep Q-Network\"\"\"\n",
            "    \n",
            "    def __init__(self, state_dim, action_dim, hidden_dim=128):\n",
            "        super().__init__()\n",
            "        \n",
            "        self.network = nn.Sequential(\n",
            "            nn.Linear(state_dim, hidden_dim),\n",
            "            nn.ReLU(),\n",
            "            nn.Linear(hidden_dim, hidden_dim),\n",
            "            nn.ReLU(),\n",
            "            nn.Linear(hidden_dim, action_dim)\n",
            "        )\n",
            "    \n",
            "    def forward(self, x):\n",
            "        return self.network(x)\n",
            "\n",
            "# Тест\n",
            "test_dqn = DQN(state_dim=4, action_dim=2)\n",
            "test_input = torch.randn(1, 4)\n",
            "print(f'Input: {test_input.shape}')\n",
            "print(f'Output: {test_dqn(test_input).shape}')\n",
            "print(f'Q-values: {test_dqn(test_input).detach().numpy()}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. DQN Agent"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "class DQNAgent:\n",
            "    \"\"\"DQN Agent с Experience Replay и Target Network\"\"\"\n",
            "    \n",
            "    def __init__(self, state_dim, action_dim, hidden_dim=128,\n",
            "                 lr=1e-3, gamma=0.99, epsilon_start=1.0, \n",
            "                 epsilon_end=0.01, epsilon_decay=0.995,\n",
            "                 buffer_size=10000, batch_size=64,\n",
            "                 target_update=10):\n",
            "        \n",
            "        self.state_dim = state_dim\n",
            "        self.action_dim = action_dim\n",
            "        self.gamma = gamma\n",
            "        self.batch_size = batch_size\n",
            "        self.target_update = target_update\n",
            "        \n",
            "        # Epsilon для exploration\n",
            "        self.epsilon = epsilon_start\n",
            "        self.epsilon_end = epsilon_end\n",
            "        self.epsilon_decay = epsilon_decay\n",
            "        \n",
            "        # Сети\n",
            "        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(device)\n",
            "        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(device)\n",
            "        self.target_net.load_state_dict(self.policy_net.state_dict())\n",
            "        \n",
            "        # Оптимизатор и буфер\n",
            "        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)\n",
            "        self.buffer = ReplayBuffer(buffer_size)\n",
            "        \n",
            "        self.steps = 0\n",
            "    \n",
            "    def select_action(self, state):\n",
            "        \"\"\"Epsilon-greedy выбор действия\"\"\"\n",
            "        if np.random.random() < self.epsilon:\n",
            "            return np.random.randint(self.action_dim)\n",
            "        \n",
            "        with torch.no_grad():\n",
            "            state = torch.FloatTensor(state).unsqueeze(0).to(device)\n",
            "            q_values = self.policy_net(state)\n",
            "            return q_values.argmax().item()\n",
            "    \n",
            "    def store_transition(self, state, action, next_state, reward, done):\n",
            "        self.buffer.push(state, action, next_state, reward, done)\n",
            "    \n",
            "    def update(self):\n",
            "        \"\"\"Обновление policy network\"\"\"\n",
            "        if len(self.buffer) < self.batch_size:\n",
            "            return 0\n",
            "        \n",
            "        # Сэмплируем батч\n",
            "        batch = self.buffer.sample(self.batch_size)\n",
            "        \n",
            "        states = torch.FloatTensor(np.array(batch.state)).to(device)\n",
            "        actions = torch.LongTensor(batch.action).to(device)\n",
            "        next_states = torch.FloatTensor(np.array(batch.next_state)).to(device)\n",
            "        rewards = torch.FloatTensor(batch.reward).to(device)\n",
            "        dones = torch.FloatTensor(batch.done).to(device)\n",
            "        \n",
            "        # Текущие Q-values\n",
            "        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()\n",
            "        \n",
            "        # Target Q-values\n",
            "        with torch.no_grad():\n",
            "            next_q = self.target_net(next_states).max(1)[0]\n",
            "            target_q = rewards + self.gamma * next_q * (1 - dones)\n",
            "        \n",
            "        # Loss и оптимизация\n",
            "        loss = nn.MSELoss()(current_q, target_q)\n",
            "        \n",
            "        self.optimizer.zero_grad()\n",
            "        loss.backward()\n",
            "        self.optimizer.step()\n",
            "        \n",
            "        # Обновление epsilon\n",
            "        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)\n",
            "        \n",
            "        # Обновление target network\n",
            "        self.steps += 1\n",
            "        if self.steps % self.target_update == 0:\n",
            "            self.target_net.load_state_dict(self.policy_net.state_dict())\n",
            "        \n",
            "        return loss.item()\n",
            "\n",
            "print('DQNAgent создан')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. CartPole Environment (симуляция)"
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
            "        # Пределы\n",
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
            "        \n",
            "        costheta = np.cos(theta)\n",
            "        sintheta = np.sin(theta)\n",
            "        \n",
            "        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass\n",
            "        thetaacc = (self.gravity * sintheta - costheta * temp) / \\\n",
            "                   (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))\n",
            "        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass\n",
            "        \n",
            "        # Euler integration\n",
            "        x = x + self.tau * x_dot\n",
            "        x_dot = x_dot + self.tau * xacc\n",
            "        theta = theta + self.tau * theta_dot\n",
            "        theta_dot = theta_dot + self.tau * thetaacc\n",
            "        \n",
            "        self.state = np.array([x, x_dot, theta, theta_dot])\n",
            "        self.steps_count += 1\n",
            "        \n",
            "        # Проверка окончания\n",
            "        done = bool(\n",
            "            x < -self.x_threshold or x > self.x_threshold or\n",
            "            theta < -self.theta_threshold or theta > self.theta_threshold or\n",
            "            self.steps_count >= 500\n",
            "        )\n",
            "        \n",
            "        reward = 1.0 if not done else 0.0\n",
            "        \n",
            "        return self.state.copy(), reward, done\n",
            "\n",
            "env = SimpleCartPole()\n",
            "state = env.reset()\n",
            "print(f'State shape: {state.shape}')\n",
            "print(f'Initial state: {state}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Обучение DQN на CartPole"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def train_dqn(env, agent, episodes=300):\n",
            "    \"\"\"Обучение DQN агента\"\"\"\n",
            "    \n",
            "    rewards_history = []\n",
            "    losses_history = []\n",
            "    \n",
            "    for episode in range(episodes):\n",
            "        state = env.reset()\n",
            "        total_reward = 0\n",
            "        episode_losses = []\n",
            "        \n",
            "        done = False\n",
            "        while not done:\n",
            "            # Выбор действия\n",
            "            action = agent.select_action(state)\n",
            "            \n",
            "            # Шаг в среде\n",
            "            next_state, reward, done = env.step(action)\n",
            "            \n",
            "            # Сохранение перехода\n",
            "            agent.store_transition(state, action, next_state, reward, done)\n",
            "            \n",
            "            # Обновление агента\n",
            "            loss = agent.update()\n",
            "            if loss > 0:\n",
            "                episode_losses.append(loss)\n",
            "            \n",
            "            total_reward += reward\n",
            "            state = next_state\n",
            "        \n",
            "        rewards_history.append(total_reward)\n",
            "        if episode_losses:\n",
            "            losses_history.append(np.mean(episode_losses))\n",
            "        \n",
            "        if (episode + 1) % 50 == 0:\n",
            "            avg_reward = np.mean(rewards_history[-50:])\n",
            "            print(f'Episode {episode+1}, Avg Reward: {avg_reward:.1f}, Epsilon: {agent.epsilon:.3f}')\n",
            "    \n",
            "    return rewards_history, losses_history\n",
            "\n",
            "# Создание и обучение агента\n",
            "env = SimpleCartPole()\n",
            "agent = DQNAgent(\n",
            "    state_dim=4,\n",
            "    action_dim=2,\n",
            "    hidden_dim=64,\n",
            "    lr=1e-3,\n",
            "    gamma=0.99,\n",
            "    epsilon_decay=0.995,\n",
            "    buffer_size=10000,\n",
            "    batch_size=64,\n",
            "    target_update=10\n",
            ")\n",
            "\n",
            "print('Начинаем обучение DQN...\\n')\n",
            "rewards, losses = train_dqn(env, agent, episodes=300)"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Визуализация обучения\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n",
            "\n",
            "# Rewards\n",
            "axes[0].plot(rewards, alpha=0.3, label='Episode Reward')\n",
            "if len(rewards) >= 20:\n",
            "    smooth_rewards = np.convolve(rewards, np.ones(20)/20, mode='valid')\n",
            "    axes[0].plot(range(19, len(rewards)), smooth_rewards, linewidth=2, label='Moving Avg (20)')\n",
            "axes[0].set_xlabel('Episode')\n",
            "axes[0].set_ylabel('Total Reward')\n",
            "axes[0].set_title('DQN Training - Rewards')\n",
            "axes[0].legend()\n",
            "\n",
            "# Losses\n",
            "if losses:\n",
            "    axes[1].plot(losses, alpha=0.5)\n",
            "    axes[1].set_xlabel('Episode')\n",
            "    axes[1].set_ylabel('Loss')\n",
            "    axes[1].set_title('DQN Training - Loss')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(f'\\nФинальная средняя награда (последние 50): {np.mean(rewards[-50:]):.1f}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Тестирование обученного агента"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def test_agent(env, agent, episodes=10):\n",
            "    \"\"\"Тестирование агента без exploration\"\"\"\n",
            "    \n",
            "    test_rewards = []\n",
            "    original_epsilon = agent.epsilon\n",
            "    agent.epsilon = 0  # Без exploration\n",
            "    \n",
            "    for episode in range(episodes):\n",
            "        state = env.reset()\n",
            "        total_reward = 0\n",
            "        done = False\n",
            "        \n",
            "        while not done:\n",
            "            action = agent.select_action(state)\n",
            "            state, reward, done = env.step(action)\n",
            "            total_reward += reward\n",
            "        \n",
            "        test_rewards.append(total_reward)\n",
            "    \n",
            "    agent.epsilon = original_epsilon\n",
            "    return test_rewards\n",
            "\n",
            "# Тестирование\n",
            "test_rewards = test_agent(env, agent, episodes=20)\n",
            "\n",
            "print('Test Results:')\n",
            "print(f'Mean Reward: {np.mean(test_rewards):.1f}')\n",
            "print(f'Std Reward: {np.std(test_rewards):.1f}')\n",
            "print(f'Max Reward: {np.max(test_rewards):.1f}')\n",
            "print(f'Min Reward: {np.min(test_rewards):.1f}')"
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
            "1. **Q-Learning** - базовый алгоритм RL\n",
            "2. **DQN** - использование нейросети для Q-функции\n",
            "3. **Experience Replay** - повторное использование опыта\n",
            "4. **Target Network** - стабилизация обучения\n",
            "5. **Epsilon-greedy** - баланс exploration/exploitation\n",
            "\n",
            "### Ключевые формулы:\n",
            "\n",
            "**Q-Learning Update:**\n",
            "$$Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)]$$\n",
            "\n",
            "**DQN Loss:**\n",
            "$$L = (r + \\gamma \\max_{a'} Q_{target}(s',a') - Q_{policy}(s,a))^2$$\n",
            "\n",
            "### Следующий шаг:\n",
            "\n",
            "В ноутбуке 02 изучим Policy Gradient методы (REINFORCE, Actor-Critic)."
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
    output_path = "/home/user/test/notebooks/phase9_reinforcement_learning/01_dqn_basics.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"Notebook created: {output_path}")
    print(f"Total cells: {len(notebook['cells'])}")
