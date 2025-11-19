#!/usr/bin/env python3
"""
Скрипт для создания 01_data_collection.ipynb
Сбор и подготовка финансовых данных
"""

import json

def create_notebook():
    cells = []

    # Cell 1: Title
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# End-to-End Trading Project\n",
            "## Часть 1: Сбор и Подготовка Данных\n",
            "\n",
            "### Цель проекта\n",
            "\n",
            "Создать полный цикл разработки торговой системы:\n",
            "1. **Сбор данных** → текущий ноутбук\n",
            "2. **Feature Engineering** → технические индикаторы\n",
            "3. **Classical ML** → baseline модели\n",
            "4. **Deep Learning** → LSTM, CNN\n",
            "5. **TFT** → продвинутые трансформеры\n",
            "6. **XAI** → интерпретация сигналов\n",
            "7. **Backtesting** → тестирование стратегий\n",
            "8. **Production** → API и MLOps\n",
            "\n",
            "### В этом ноутбуке:\n",
            "\n",
            "- Генерация реалистичных данных для 10 акций (5 лет истории)\n",
            "- OHLCV данные с рыночными паттернами\n",
            "- Корреляции между акциями (секторы)\n",
            "- Базовая визуализация и статистика\n",
            "- Сохранение данных для последующих этапов"
        ]
    })

    # Cell 2: Imports
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from datetime import datetime, timedelta\n",
            "import os\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Настройки визуализации\n",
            "plt.style.use('seaborn-v0_8-whitegrid')\n",
            "plt.rcParams['figure.figsize'] = (12, 6)\n",
            "plt.rcParams['font.size'] = 10\n",
            "\n",
            "# Фиксируем seed\n",
            "np.random.seed(42)\n",
            "\n",
            "print('Библиотеки загружены')\n",
            "print(f'NumPy: {np.__version__}')\n",
            "print(f'Pandas: {pd.__version__}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 3: Stock Universe
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Определение Инвестиционной Вселенной\n",
            "\n",
            "Создаём портфель из 10 акций, разделённых на 3 сектора:\n",
            "- **Technology** (4 акции) - высокая волатильность, рост\n",
            "- **Finance** (3 акции) - средняя волатильность\n",
            "- **Consumer** (3 акции) - низкая волатильность, стабильность"
        ]
    })

    # Cell 4: Stock Definitions
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Определяем акции и их характеристики\n",
            "stocks = {\n",
            "    # Technology - высокий рост, высокая волатильность\n",
            "    'TECH_A': {'sector': 'Technology', 'base_price': 150, 'drift': 0.0003, 'volatility': 0.025},\n",
            "    'TECH_B': {'sector': 'Technology', 'base_price': 80, 'drift': 0.0004, 'volatility': 0.030},\n",
            "    'TECH_C': {'sector': 'Technology', 'base_price': 200, 'drift': 0.0002, 'volatility': 0.022},\n",
            "    'TECH_D': {'sector': 'Technology', 'base_price': 120, 'drift': 0.0005, 'volatility': 0.035},\n",
            "    \n",
            "    # Finance - средний рост, средняя волатильность\n",
            "    'FIN_A': {'sector': 'Finance', 'base_price': 100, 'drift': 0.0001, 'volatility': 0.018},\n",
            "    'FIN_B': {'sector': 'Finance', 'base_price': 60, 'drift': 0.0002, 'volatility': 0.020},\n",
            "    'FIN_C': {'sector': 'Finance', 'base_price': 90, 'drift': 0.0001, 'volatility': 0.016},\n",
            "    \n",
            "    # Consumer - стабильность, низкая волатильность\n",
            "    'CONS_A': {'sector': 'Consumer', 'base_price': 50, 'drift': 0.0001, 'volatility': 0.012},\n",
            "    'CONS_B': {'sector': 'Consumer', 'base_price': 70, 'drift': 0.0001, 'volatility': 0.014},\n",
            "    'CONS_C': {'sector': 'Consumer', 'base_price': 45, 'drift': 0.0002, 'volatility': 0.015},\n",
            "}\n",
            "\n",
            "print('Инвестиционная вселенная:')\n",
            "print(f'Всего акций: {len(stocks)}')\n",
            "print(f'Секторы: {len(set(s[\"sector\"] for s in stocks.values()))}')\n",
            "print()\n",
            "\n",
            "# Выводим таблицу\n",
            "stock_info = pd.DataFrame(stocks).T\n",
            "stock_info"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 5: Market Generator
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Генератор Рыночных Данных\n",
            "\n",
            "Создаём реалистичные данные с учётом:\n",
            "- **Рыночный фактор** - общее движение рынка влияет на все акции\n",
            "- **Секторный фактор** - акции одного сектора коррелируют\n",
            "- **Индивидуальный шум** - уникальное движение каждой акции\n",
            "- **Режимы рынка** - бычий, медвежий, боковой\n",
            "- **События** - earnings, news, crashes"
        ]
    })

    # Cell 6: Market Regime Generator
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def generate_market_regimes(n_days):\n",
            "    \"\"\"\n",
            "    Генерирует режимы рынка:\n",
            "    - Bull (бычий): положительный drift, низкая волатильность\n",
            "    - Bear (медвежий): отрицательный drift, высокая волатильность  \n",
            "    - Sideways (боковой): нулевой drift, средняя волатильность\n",
            "    \"\"\"\n",
            "    regimes = []\n",
            "    regime_params = {\n",
            "        'bull': {'drift_mult': 1.5, 'vol_mult': 0.8, 'duration': (60, 180)},\n",
            "        'bear': {'drift_mult': -1.0, 'vol_mult': 1.5, 'duration': (30, 90)},\n",
            "        'sideways': {'drift_mult': 0.2, 'vol_mult': 1.0, 'duration': (30, 120)}\n",
            "    }\n",
            "    \n",
            "    current_day = 0\n",
            "    while current_day < n_days:\n",
            "        # Выбираем режим (bull чаще, т.к. рынок растёт долгосрочно)\n",
            "        regime = np.random.choice(['bull', 'bear', 'sideways'], p=[0.5, 0.2, 0.3])\n",
            "        params = regime_params[regime]\n",
            "        \n",
            "        # Длительность режима\n",
            "        duration = np.random.randint(params['duration'][0], params['duration'][1])\n",
            "        end_day = min(current_day + duration, n_days)\n",
            "        \n",
            "        for day in range(current_day, end_day):\n",
            "            regimes.append({\n",
            "                'regime': regime,\n",
            "                'drift_mult': params['drift_mult'],\n",
            "                'vol_mult': params['vol_mult']\n",
            "            })\n",
            "        \n",
            "        current_day = end_day\n",
            "    \n",
            "    return regimes[:n_days]\n",
            "\n",
            "# Параметры генерации\n",
            "n_years = 5\n",
            "n_days = n_years * 252  # Торговых дней в году\n",
            "start_date = datetime(2019, 1, 2)\n",
            "\n",
            "# Генерируем режимы\n",
            "market_regimes = generate_market_regimes(n_days)\n",
            "\n",
            "# Статистика режимов\n",
            "regime_counts = pd.Series([r['regime'] for r in market_regimes]).value_counts()\n",
            "print(f'Всего торговых дней: {n_days}')\n",
            "print(f'\\nРаспределение режимов:')\n",
            "for regime, count in regime_counts.items():\n",
            "    print(f'  {regime}: {count} дней ({count/n_days*100:.1f}%)')"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 7: Correlated Returns Generator
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def generate_correlated_returns(stocks, n_days, market_regimes):\n",
            "    \"\"\"\n",
            "    Генерирует коррелированные доходности с учётом:\n",
            "    - Рыночного фактора (beta)\n",
            "    - Секторного фактора\n",
            "    - Индивидуального шума\n",
            "    \"\"\"\n",
            "    n_stocks = len(stocks)\n",
            "    stock_names = list(stocks.keys())\n",
            "    \n",
            "    # Рыночный фактор\n",
            "    market_returns = np.zeros(n_days)\n",
            "    for i, regime in enumerate(market_regimes):\n",
            "        market_returns[i] = np.random.normal(0.0003 * regime['drift_mult'], \n",
            "                                            0.01 * regime['vol_mult'])\n",
            "    \n",
            "    # Секторные факторы\n",
            "    sectors = list(set(s['sector'] for s in stocks.values()))\n",
            "    sector_returns = {}\n",
            "    for sector in sectors:\n",
            "        sector_returns[sector] = np.random.normal(0, 0.005, n_days)\n",
            "    \n",
            "    # Генерируем доходности для каждой акции\n",
            "    returns = {}\n",
            "    \n",
            "    for name, params in stocks.items():\n",
            "        # Компоненты доходности\n",
            "        beta = np.random.uniform(0.8, 1.2)  # Чувствительность к рынку\n",
            "        \n",
            "        stock_returns = np.zeros(n_days)\n",
            "        for i, regime in enumerate(market_regimes):\n",
            "            # Drift с учётом режима\n",
            "            drift = params['drift'] * regime['drift_mult']\n",
            "            \n",
            "            # Волатильность с учётом режима\n",
            "            vol = params['volatility'] * regime['vol_mult']\n",
            "            \n",
            "            # Итоговая доходность = drift + market + sector + noise\n",
            "            market_component = beta * market_returns[i] * 0.6\n",
            "            sector_component = sector_returns[params['sector']][i] * 0.3\n",
            "            idiosyncratic = np.random.normal(0, vol) * 0.5\n",
            "            \n",
            "            stock_returns[i] = drift + market_component + sector_component + idiosyncratic\n",
            "        \n",
            "        returns[name] = stock_returns\n",
            "    \n",
            "    return returns, market_returns\n",
            "\n",
            "# Генерируем доходности\n",
            "stock_returns, market_returns = generate_correlated_returns(stocks, n_days, market_regimes)\n",
            "\n",
            "print('Доходности сгенерированы')\n",
            "print(f'\\nПример статистики (TECH_A):')\n",
            "print(f'  Mean daily return: {np.mean(stock_returns[\"TECH_A\"])*100:.4f}%')\n",
            "print(f'  Std daily return: {np.std(stock_returns[\"TECH_A\"])*100:.4f}%')\n",
            "print(f'  Annualized return: {np.mean(stock_returns[\"TECH_A\"])*252*100:.2f}%')\n",
            "print(f'  Annualized volatility: {np.std(stock_returns[\"TECH_A\"])*np.sqrt(252)*100:.2f}%')"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 8: OHLCV Generator
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def generate_ohlcv(stock_name, params, returns, start_date, n_days):\n",
            "    \"\"\"\n",
            "    Генерирует OHLCV данные из доходностей.\n",
            "    \"\"\"\n",
            "    # Генерируем цены закрытия из доходностей\n",
            "    prices = [params['base_price']]\n",
            "    for r in returns:\n",
            "        prices.append(prices[-1] * (1 + r))\n",
            "    closes = np.array(prices[1:])  # Убираем начальную цену\n",
            "    \n",
            "    # Генерируем Open, High, Low\n",
            "    # Open = Close предыдущего дня + небольшой gap\n",
            "    opens = np.roll(closes, 1) * (1 + np.random.normal(0, 0.002, n_days))\n",
            "    opens[0] = params['base_price']\n",
            "    \n",
            "    # High и Low\n",
            "    daily_range = np.abs(returns) + 0.005  # Минимальный диапазон\n",
            "    highs = np.maximum(opens, closes) * (1 + daily_range * np.random.uniform(0.3, 1.0, n_days))\n",
            "    lows = np.minimum(opens, closes) * (1 - daily_range * np.random.uniform(0.3, 1.0, n_days))\n",
            "    \n",
            "    # Volume - коррелирует с волатильностью и трендом\n",
            "    base_volume = np.random.uniform(1e6, 5e6)\n",
            "    vol_factor = 1 + 10 * np.abs(returns)  # Больше объём при больших движениях\n",
            "    volumes = base_volume * vol_factor * np.random.uniform(0.7, 1.3, n_days)\n",
            "    \n",
            "    # Создаём даты (только торговые дни)\n",
            "    dates = pd.bdate_range(start=start_date, periods=n_days)\n",
            "    \n",
            "    # DataFrame\n",
            "    df = pd.DataFrame({\n",
            "        'date': dates,\n",
            "        'ticker': stock_name,\n",
            "        'open': opens,\n",
            "        'high': highs,\n",
            "        'low': lows,\n",
            "        'close': closes,\n",
            "        'volume': volumes.astype(int),\n",
            "        'sector': params['sector']\n",
            "    })\n",
            "    \n",
            "    return df\n",
            "\n",
            "# Генерируем данные для всех акций\n",
            "all_data = []\n",
            "for name, params in stocks.items():\n",
            "    df = generate_ohlcv(name, params, stock_returns[name], start_date, n_days)\n",
            "    all_data.append(df)\n",
            "    \n",
            "# Объединяем\n",
            "market_data = pd.concat(all_data, ignore_index=True)\n",
            "\n",
            "print(f'Размер датасета: {len(market_data):,} записей')\n",
            "print(f'Период: {market_data[\"date\"].min().date()} - {market_data[\"date\"].max().date()}')\n",
            "print(f'\\nПервые записи:')\n",
            "market_data.head(10)"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 9: Add Events
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def add_market_events(df):\n",
            "    \"\"\"\n",
            "    Добавляет рыночные события:\n",
            "    - Earnings (квартальные отчёты)\n",
            "    - Dividends\n",
            "    - Splits\n",
            "    - News events\n",
            "    \"\"\"\n",
            "    df = df.copy()\n",
            "    \n",
            "    # Инициализируем колонки событий\n",
            "    df['is_earnings'] = False\n",
            "    df['is_dividend'] = False\n",
            "    df['event_impact'] = 0.0\n",
            "    \n",
            "    for ticker in df['ticker'].unique():\n",
            "        mask = df['ticker'] == ticker\n",
            "        ticker_dates = df.loc[mask, 'date'].values\n",
            "        n_days = len(ticker_dates)\n",
            "        \n",
            "        # Earnings - каждые ~63 дня (квартал)\n",
            "        earnings_days = np.arange(60, n_days, 63)\n",
            "        for day in earnings_days:\n",
            "            idx = df[(df['ticker'] == ticker)].index[day]\n",
            "            df.loc[idx, 'is_earnings'] = True\n",
            "            # Impact: -5% to +10%\n",
            "            impact = np.random.uniform(-0.05, 0.10)\n",
            "            df.loc[idx, 'event_impact'] = impact\n",
            "        \n",
            "        # Dividends - каждые ~126 дней (полугодие)\n",
            "        dividend_days = np.arange(120, n_days, 126)\n",
            "        for day in dividend_days:\n",
            "            idx = df[(df['ticker'] == ticker)].index[day]\n",
            "            df.loc[idx, 'is_dividend'] = True\n",
            "    \n",
            "    return df\n",
            "\n",
            "# Добавляем события\n",
            "market_data = add_market_events(market_data)\n",
            "\n",
            "print('События добавлены:')\n",
            "print(f'Earnings events: {market_data[\"is_earnings\"].sum()}')\n",
            "print(f'Dividend events: {market_data[\"is_dividend\"].sum()}')\n",
            "print(f'\\nПример событий:')\n",
            "market_data[market_data['is_earnings']].head()"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 10: Basic Stats
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Базовая Статистика и Визуализация"
        ]
    })

    # Cell 11: Summary Stats
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Рассчитываем статистику для каждой акции\n",
            "summary_stats = []\n",
            "\n",
            "for ticker in market_data['ticker'].unique():\n",
            "    ticker_data = market_data[market_data['ticker'] == ticker].copy()\n",
            "    ticker_data['return'] = ticker_data['close'].pct_change()\n",
            "    \n",
            "    stats = {\n",
            "        'ticker': ticker,\n",
            "        'sector': ticker_data['sector'].iloc[0],\n",
            "        'start_price': ticker_data['close'].iloc[0],\n",
            "        'end_price': ticker_data['close'].iloc[-1],\n",
            "        'total_return': (ticker_data['close'].iloc[-1] / ticker_data['close'].iloc[0] - 1) * 100,\n",
            "        'annual_return': ((ticker_data['close'].iloc[-1] / ticker_data['close'].iloc[0]) ** (252/len(ticker_data)) - 1) * 100,\n",
            "        'annual_volatility': ticker_data['return'].std() * np.sqrt(252) * 100,\n",
            "        'sharpe_ratio': (ticker_data['return'].mean() * 252) / (ticker_data['return'].std() * np.sqrt(252)),\n",
            "        'max_drawdown': ((ticker_data['close'] / ticker_data['close'].cummax() - 1).min()) * 100,\n",
            "        'avg_volume': ticker_data['volume'].mean() / 1e6\n",
            "    }\n",
            "    summary_stats.append(stats)\n",
            "\n",
            "summary_df = pd.DataFrame(summary_stats)\n",
            "summary_df = summary_df.round(2)\n",
            "\n",
            "print('Сводная статистика по акциям:\\n')\n",
            "print(summary_df.to_string(index=False))"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 12: Price Charts
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Визуализация цен\n",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
            "\n",
            "# 1. Все акции (нормализованные)\n",
            "ax = axes[0, 0]\n",
            "for ticker in market_data['ticker'].unique():\n",
            "    ticker_data = market_data[market_data['ticker'] == ticker]\n",
            "    normalized = ticker_data['close'] / ticker_data['close'].iloc[0] * 100\n",
            "    ax.plot(ticker_data['date'], normalized, label=ticker, alpha=0.7)\n",
            "ax.set_title('Нормализованные цены (база = 100)')\n",
            "ax.set_xlabel('Дата')\n",
            "ax.set_ylabel('Индекс')\n",
            "ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)\n",
            "\n",
            "# 2. По секторам\n",
            "ax = axes[0, 1]\n",
            "sector_performance = {}\n",
            "for sector in market_data['sector'].unique():\n",
            "    sector_data = market_data[market_data['sector'] == sector]\n",
            "    # Средняя нормализованная цена по сектору\n",
            "    sector_prices = []\n",
            "    for ticker in sector_data['ticker'].unique():\n",
            "        ticker_data = sector_data[sector_data['ticker'] == ticker]\n",
            "        normalized = ticker_data['close'] / ticker_data['close'].iloc[0]\n",
            "        sector_prices.append(normalized.values)\n",
            "    mean_price = np.mean(sector_prices, axis=0) * 100\n",
            "    dates = sector_data[sector_data['ticker'] == ticker]['date']\n",
            "    ax.plot(dates, mean_price, label=sector, linewidth=2)\n",
            "ax.set_title('Средняя доходность по секторам')\n",
            "ax.set_xlabel('Дата')\n",
            "ax.set_ylabel('Индекс')\n",
            "ax.legend()\n",
            "\n",
            "# 3. Распределение доходностей\n",
            "ax = axes[1, 0]\n",
            "all_returns = []\n",
            "for ticker in market_data['ticker'].unique():\n",
            "    ticker_data = market_data[market_data['ticker'] == ticker]\n",
            "    returns = ticker_data['close'].pct_change().dropna()\n",
            "    all_returns.extend(returns)\n",
            "ax.hist(all_returns, bins=100, edgecolor='black', alpha=0.7)\n",
            "ax.axvline(x=0, color='red', linestyle='--')\n",
            "ax.set_title('Распределение дневных доходностей')\n",
            "ax.set_xlabel('Доходность')\n",
            "ax.set_ylabel('Частота')\n",
            "\n",
            "# 4. Объёмы\n",
            "ax = axes[1, 1]\n",
            "tech_a = market_data[market_data['ticker'] == 'TECH_A'].tail(252)  # Последний год\n",
            "ax.bar(tech_a['date'], tech_a['volume'] / 1e6, alpha=0.7, width=1)\n",
            "ax.set_title('Объём торгов TECH_A (последний год)')\n",
            "ax.set_xlabel('Дата')\n",
            "ax.set_ylabel('Объём (млн)')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 13: Correlation
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Матрица корреляций доходностей\n",
            "returns_matrix = pd.DataFrame()\n",
            "\n",
            "for ticker in market_data['ticker'].unique():\n",
            "    ticker_data = market_data[market_data['ticker'] == ticker].set_index('date')\n",
            "    returns_matrix[ticker] = ticker_data['close'].pct_change()\n",
            "\n",
            "correlation_matrix = returns_matrix.corr()\n",
            "\n",
            "# Визуализация\n",
            "fig, ax = plt.subplots(figsize=(10, 8))\n",
            "sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0,\n",
            "            fmt='.2f', square=True, ax=ax)\n",
            "ax.set_title('Корреляция доходностей между акциями')\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print('\\nСредняя корреляция по секторам:')\n",
            "for sector in ['Technology', 'Finance', 'Consumer']:\n",
            "    sector_tickers = [t for t, p in stocks.items() if p['sector'] == sector]\n",
            "    sector_corr = correlation_matrix.loc[sector_tickers, sector_tickers]\n",
            "    # Убираем диагональ\n",
            "    mask = np.triu(np.ones_like(sector_corr, dtype=bool), k=1)\n",
            "    avg_corr = sector_corr.where(mask).stack().mean()\n",
            "    print(f'  {sector}: {avg_corr:.3f}')"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 14: Volatility Clustering
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Демонстрация кластеризации волатильности\n",
            "tech_a = market_data[market_data['ticker'] == 'TECH_A'].copy()\n",
            "tech_a['return'] = tech_a['close'].pct_change()\n",
            "tech_a['volatility_20d'] = tech_a['return'].rolling(20).std() * np.sqrt(252)\n",
            "\n",
            "fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)\n",
            "\n",
            "# Цена\n",
            "axes[0].plot(tech_a['date'], tech_a['close'], linewidth=1)\n",
            "axes[0].set_ylabel('Цена')\n",
            "axes[0].set_title('TECH_A: Цена и Волатильность')\n",
            "\n",
            "# Волатильность\n",
            "axes[1].fill_between(tech_a['date'], 0, tech_a['volatility_20d'], alpha=0.5)\n",
            "axes[1].axhline(y=tech_a['volatility_20d'].mean(), color='red', linestyle='--', \n",
            "                label=f'Mean: {tech_a[\"volatility_20d\"].mean():.2%}')\n",
            "axes[1].set_ylabel('20-дневная волатильность (годовая)')\n",
            "axes[1].set_xlabel('Дата')\n",
            "axes[1].legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print('Кластеризация волатильности - периоды высокой волатильности группируются вместе')\n",
            "print('Это важное свойство финансовых данных (GARCH-эффект)')"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 15: Save Data
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Сохранение Данных\n",
            "\n",
            "Сохраняем данные для использования в последующих ноутбуках."
        ]
    })

    # Cell 16: Save
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Создаём директорию для данных\n",
            "data_dir = 'data'\n",
            "os.makedirs(data_dir, exist_ok=True)\n",
            "\n",
            "# Сохраняем основные данные\n",
            "market_data.to_parquet(f'{data_dir}/market_data.parquet', index=False)\n",
            "market_data.to_csv(f'{data_dir}/market_data.csv', index=False)\n",
            "\n",
            "# Сохраняем метаданные\n",
            "import json\n",
            "metadata = {\n",
            "    'stocks': stocks,\n",
            "    'n_days': n_days,\n",
            "    'start_date': str(start_date.date()),\n",
            "    'end_date': str(market_data['date'].max().date()),\n",
            "    'tickers': list(stocks.keys()),\n",
            "    'sectors': list(set(s['sector'] for s in stocks.values()))\n",
            "}\n",
            "\n",
            "with open(f'{data_dir}/metadata.json', 'w') as f:\n",
            "    json.dump(metadata, f, indent=2, default=str)\n",
            "\n",
            "print('Данные сохранены:')\n",
            "print(f'  {data_dir}/market_data.parquet')\n",
            "print(f'  {data_dir}/market_data.csv')\n",
            "print(f'  {data_dir}/metadata.json')\n",
            "print(f'\\nРазмер файлов:')\n",
            "for f in os.listdir(data_dir):\n",
            "    size = os.path.getsize(f'{data_dir}/{f}') / 1024\n",
            "    print(f'  {f}: {size:.1f} KB')"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 17: Summary
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Итоги\n",
            "\n",
            "### Созданные данные:\n",
            "\n",
            "- **10 акций** из 3 секторов (Technology, Finance, Consumer)\n",
            "- **5 лет** торговой истории (~1260 дней на акцию)\n",
            "- **12,600 записей** OHLCV данных\n",
            "- Реалистичные корреляции между акциями одного сектора\n",
            "- Рыночные режимы (bull/bear/sideways)\n",
            "- События (earnings, dividends)\n",
            "\n",
            "### Следующий шаг:\n",
            "\n",
            "В следующем ноутбуке (02_feature_engineering) мы:\n",
            "- Добавим технические индикаторы (SMA, EMA, RSI, MACD, Bollinger Bands)\n",
            "- Создадим лаговые признаки\n",
            "- Проведём подробный EDA\n",
            "- Подготовим данные для моделирования"
        ]
    })

    # Create notebook
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
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
        "cells": cells
    }

    return notebook

if __name__ == "__main__":
    notebook = create_notebook()
    output_path = "/home/user/test/notebooks/end_to_end_trading/01_data_collection.ipynb"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"Notebook created: {output_path}")
    print(f"Total cells: {len(notebook['cells'])}")
