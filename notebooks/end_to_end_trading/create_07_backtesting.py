#!/usr/bin/env python3
"""
Скрипт для создания 07_backtesting.ipynb
Бэктестинг торговых стратегий
"""

import json

def create_notebook():
    cells = []

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# End-to-End Trading Project\n",
            "## Часть 7: Бэктестинг Стратегий\n",
            "\n",
            "### В этом ноутбуке:\n",
            "\n",
            "1. **Создание торговых стратегий** на основе ML моделей\n",
            "2. **Бэктестинг** с учётом комиссий и slippage\n",
            "3. **Метрики**: Sharpe, Max Drawdown, Win Rate\n",
            "4. **Сравнение** с benchmark (Buy & Hold)"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import joblib\n",
            "import json\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "plt.style.use('seaborn-v0_8-whitegrid')\n",
            "np.random.seed(42)\n",
            "print('Библиотеки загружены')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Загружаем данные и модель\n",
            "data_dir = 'data'\n",
            "models_dir = 'models'\n",
            "\n",
            "df = pd.read_parquet(f'{data_dir}/processed_data.parquet')\n",
            "with open(f'{data_dir}/feature_sets.json', 'r') as f:\n",
            "    feature_sets = json.load(f)\n",
            "\n",
            "lgb_model = joblib.load(f'{models_dir}/lightgbm.joblib')\n",
            "\n",
            "feature_cols = [f for f in feature_sets['extended_features'] if f in df.columns]\n",
            "\n",
            "# Тестовый период\n",
            "df = df.sort_values('date')\n",
            "test_df = df.iloc[int(len(df)*0.8):].copy()\n",
            "print(f'Тестовый период: {test_df[\"date\"].min().date()} - {test_df[\"date\"].max().date()}')\n",
            "print(f'Записей: {len(test_df):,}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Генерация Торговых Сигналов"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Генерируем сигналы для каждой акции\n",
            "signals = []\n",
            "\n",
            "for ticker in test_df['ticker'].unique():\n",
            "    ticker_df = test_df[test_df['ticker'] == ticker].copy()\n",
            "    \n",
            "    X = ticker_df[feature_cols].values\n",
            "    proba = lgb_model.predict_proba(X)[:, 1]\n",
            "    \n",
            "    ticker_df['signal_proba'] = proba\n",
            "    ticker_df['signal'] = 0\n",
            "    ticker_df.loc[ticker_df['signal_proba'] > 0.55, 'signal'] = 1  # Buy\n",
            "    ticker_df.loc[ticker_df['signal_proba'] < 0.45, 'signal'] = -1  # Sell\n",
            "    \n",
            "    signals.append(ticker_df)\n",
            "\n",
            "signals_df = pd.concat(signals)\n",
            "\n",
            "print('Распределение сигналов:')\n",
            "print(signals_df['signal'].value_counts())"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Бэктестинг Engine"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "class SimpleBacktester:\n",
            "    \"\"\"\n",
            "    Простой бэктестер для торговых стратегий.\n",
            "    \"\"\"\n",
            "    def __init__(self, commission=0.001, slippage=0.0005):\n",
            "        self.commission = commission\n",
            "        self.slippage = slippage\n",
            "    \n",
            "    def run(self, df, signal_col='signal', price_col='close'):\n",
            "        \"\"\"\n",
            "        Запуск бэктеста.\n",
            "        signal: 1 (long), 0 (no position), -1 (short)\n",
            "        \"\"\"\n",
            "        df = df.sort_values('date').copy()\n",
            "        \n",
            "        # Рассчитываем доходности\n",
            "        df['returns'] = df[price_col].pct_change()\n",
            "        \n",
            "        # Позиция на предыдущий день (чтобы избежать look-ahead bias)\n",
            "        df['position'] = df[signal_col].shift(1).fillna(0)\n",
            "        \n",
            "        # Доходность стратегии\n",
            "        df['strategy_returns'] = df['position'] * df['returns']\n",
            "        \n",
            "        # Комиссии при изменении позиции\n",
            "        df['position_change'] = df['position'].diff().abs().fillna(0)\n",
            "        df['costs'] = df['position_change'] * (self.commission + self.slippage)\n",
            "        df['strategy_returns'] = df['strategy_returns'] - df['costs']\n",
            "        \n",
            "        # Кумулятивная доходность\n",
            "        df['cumulative_returns'] = (1 + df['returns']).cumprod()\n",
            "        df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()\n",
            "        \n",
            "        return df\n",
            "    \n",
            "    def calculate_metrics(self, df):\n",
            "        \"\"\"\n",
            "        Рассчитывает метрики производительности.\n",
            "        \"\"\"\n",
            "        strategy_returns = df['strategy_returns'].dropna()\n",
            "        \n",
            "        # Годовые метрики\n",
            "        annual_return = strategy_returns.mean() * 252\n",
            "        annual_vol = strategy_returns.std() * np.sqrt(252)\n",
            "        sharpe = annual_return / (annual_vol + 1e-10)\n",
            "        \n",
            "        # Max Drawdown\n",
            "        cumulative = (1 + strategy_returns).cumprod()\n",
            "        running_max = cumulative.cummax()\n",
            "        drawdown = (cumulative - running_max) / running_max\n",
            "        max_drawdown = drawdown.min()\n",
            "        \n",
            "        # Win rate\n",
            "        trades = strategy_returns[strategy_returns != 0]\n",
            "        win_rate = (trades > 0).mean() if len(trades) > 0 else 0\n",
            "        \n",
            "        # Profit factor\n",
            "        gross_profit = trades[trades > 0].sum()\n",
            "        gross_loss = abs(trades[trades < 0].sum())\n",
            "        profit_factor = gross_profit / (gross_loss + 1e-10)\n",
            "        \n",
            "        return {\n",
            "            'total_return': (df['cumulative_strategy'].iloc[-1] - 1) * 100,\n",
            "            'annual_return': annual_return * 100,\n",
            "            'annual_volatility': annual_vol * 100,\n",
            "            'sharpe_ratio': sharpe,\n",
            "            'max_drawdown': max_drawdown * 100,\n",
            "            'win_rate': win_rate * 100,\n",
            "            'profit_factor': profit_factor,\n",
            "            'num_trades': len(trades)\n",
            "        }\n",
            "\n",
            "backtester = SimpleBacktester(commission=0.001, slippage=0.0005)"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Бэктест для одной акции\n",
            "ticker = 'TECH_A'\n",
            "ticker_signals = signals_df[signals_df['ticker'] == ticker]\n",
            "\n",
            "results_df = backtester.run(ticker_signals)\n",
            "metrics = backtester.calculate_metrics(results_df)\n",
            "\n",
            "print(f'Результаты для {ticker}:\\n')\n",
            "for key, value in metrics.items():\n",
            "    if 'return' in key or 'volatility' in key or 'drawdown' in key or 'rate' in key:\n",
            "        print(f'  {key}: {value:.2f}%')\n",
            "    elif key == 'sharpe_ratio' or key == 'profit_factor':\n",
            "        print(f'  {key}: {value:.3f}')\n",
            "    else:\n",
            "        print(f'  {key}: {value}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Визуализация\n",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
            "\n",
            "# 1. Equity curve\n",
            "axes[0, 0].plot(results_df['date'], results_df['cumulative_returns'], label='Buy & Hold')\n",
            "axes[0, 0].plot(results_df['date'], results_df['cumulative_strategy'], label='Strategy')\n",
            "axes[0, 0].set_xlabel('Date')\n",
            "axes[0, 0].set_ylabel('Cumulative Return')\n",
            "axes[0, 0].set_title(f'{ticker} - Equity Curve')\n",
            "axes[0, 0].legend()\n",
            "\n",
            "# 2. Drawdown\n",
            "cumulative = (1 + results_df['strategy_returns']).cumprod()\n",
            "drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()\n",
            "axes[0, 1].fill_between(results_df['date'], drawdown * 100, 0, alpha=0.5, color='red')\n",
            "axes[0, 1].set_xlabel('Date')\n",
            "axes[0, 1].set_ylabel('Drawdown (%)')\n",
            "axes[0, 1].set_title('Drawdown')\n",
            "\n",
            "# 3. Signal distribution\n",
            "signal_counts = results_df['signal'].value_counts().sort_index()\n",
            "axes[1, 0].bar(signal_counts.index, signal_counts.values)\n",
            "axes[1, 0].set_xlabel('Signal')\n",
            "axes[1, 0].set_ylabel('Count')\n",
            "axes[1, 0].set_title('Signal Distribution')\n",
            "axes[1, 0].set_xticks([-1, 0, 1])\n",
            "axes[1, 0].set_xticklabels(['Sell', 'Hold', 'Buy'])\n",
            "\n",
            "# 4. Returns distribution\n",
            "returns = results_df['strategy_returns'].dropna()\n",
            "axes[1, 1].hist(returns * 100, bins=50, edgecolor='black', alpha=0.7)\n",
            "axes[1, 1].axvline(x=0, color='red', linestyle='--')\n",
            "axes[1, 1].set_xlabel('Daily Return (%)')\n",
            "axes[1, 1].set_ylabel('Count')\n",
            "axes[1, 1].set_title('Strategy Returns Distribution')\n",
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
            "## 3. Бэктест для всех акций"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Бэктест для всех акций\n",
            "all_metrics = []\n",
            "\n",
            "for ticker in signals_df['ticker'].unique():\n",
            "    ticker_signals = signals_df[signals_df['ticker'] == ticker]\n",
            "    results_df = backtester.run(ticker_signals)\n",
            "    metrics = backtester.calculate_metrics(results_df)\n",
            "    metrics['ticker'] = ticker\n",
            "    all_metrics.append(metrics)\n",
            "\n",
            "metrics_df = pd.DataFrame(all_metrics)\n",
            "print('Результаты по всем акциям:\\n')\n",
            "print(metrics_df[['ticker', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']].round(2).to_string(index=False))\n",
            "\n",
            "print(f'\\nСредние показатели:')\n",
            "print(f'  Total Return: {metrics_df[\"total_return\"].mean():.2f}%')\n",
            "print(f'  Sharpe Ratio: {metrics_df[\"sharpe_ratio\"].mean():.3f}')\n",
            "print(f'  Max Drawdown: {metrics_df[\"max_drawdown\"].mean():.2f}%')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Портфельная Стратегия"
        ]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Равновзвешенный портфель\n",
            "portfolio_returns = []\n",
            "benchmark_returns = []\n",
            "\n",
            "dates = signals_df['date'].unique()\n",
            "\n",
            "for date in sorted(dates):\n",
            "    day_data = signals_df[signals_df['date'] == date]\n",
            "    \n",
            "    # Средняя доходность стратегии по всем акциям\n",
            "    day_returns = []\n",
            "    bench_returns = []\n",
            "    \n",
            "    for ticker in day_data['ticker'].unique():\n",
            "        ticker_day = day_data[day_data['ticker'] == ticker]\n",
            "        if len(ticker_day) > 0:\n",
            "            ret = ticker_day['close'].pct_change().values[-1] if len(ticker_day) > 1 else 0\n",
            "            signal = ticker_day['signal'].values[-1]\n",
            "            \n",
            "            day_returns.append(signal * ret)\n",
            "            bench_returns.append(ret)\n",
            "    \n",
            "    if day_returns:\n",
            "        portfolio_returns.append(np.mean(day_returns))\n",
            "        benchmark_returns.append(np.mean(bench_returns))\n",
            "\n",
            "portfolio_returns = np.array(portfolio_returns)\n",
            "benchmark_returns = np.array(benchmark_returns)\n",
            "\n",
            "# Рассчитываем метрики портфеля\n",
            "portfolio_cumulative = (1 + portfolio_returns).cumprod()\n",
            "benchmark_cumulative = (1 + benchmark_returns).cumprod()\n",
            "\n",
            "print('Портфельные результаты:')\n",
            "print(f'  Portfolio Total Return: {(portfolio_cumulative[-1] - 1) * 100:.2f}%')\n",
            "print(f'  Benchmark Total Return: {(benchmark_cumulative[-1] - 1) * 100:.2f}%')\n",
            "print(f'  Portfolio Sharpe: {portfolio_returns.mean() * 252 / (portfolio_returns.std() * np.sqrt(252)):.3f}')\n",
            "print(f'  Benchmark Sharpe: {benchmark_returns.mean() * 252 / (benchmark_returns.std() * np.sqrt(252)):.3f}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Визуализация портфеля\n",
            "fig, ax = plt.subplots(figsize=(12, 6))\n",
            "ax.plot(portfolio_cumulative, label='ML Strategy Portfolio', linewidth=2)\n",
            "ax.plot(benchmark_cumulative, label='Buy & Hold Portfolio', linewidth=2, alpha=0.7)\n",
            "ax.set_xlabel('Trading Days')\n",
            "ax.set_ylabel('Cumulative Return')\n",
            "ax.set_title('Portfolio Performance: ML Strategy vs Buy & Hold')\n",
            "ax.legend()\n",
            "ax.grid(True, alpha=0.3)\n",
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
            "## Итоги\n",
            "\n",
            "### Результаты бэктеста:\n",
            "\n",
            "- Стратегия показывает смешанные результаты по акциям\n",
            "- Sharpe Ratio близок к 0 (типично для направленных стратегий)\n",
            "- Win Rate около 50% (соответствует accuracy модели)\n",
            "\n",
            "### Выводы:\n",
            "\n",
            "1. **ML модели не дают гарантированного edge** на эффективном рынке\n",
            "2. **Комиссии и slippage** существенно влияют на результат\n",
            "3. **Risk management** критичен (max drawdown)\n",
            "\n",
            "### Рекомендации:\n",
            "\n",
            "- Использовать только сигналы с высокой уверенностью\n",
            "- Добавить stop-loss и take-profit\n",
            "- Комбинировать с фундаментальным анализом\n",
            "- Регулярно переобучать модели\n",
            "\n",
            "### Следующий шаг:\n",
            "\n",
            "В ноутбуке 08 создадим production-ready систему."
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
    output_path = "/home/user/test/notebooks/end_to_end_trading/07_backtesting.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"Notebook created: {output_path}")
    print(f"Total cells: {len(notebook['cells'])}")
