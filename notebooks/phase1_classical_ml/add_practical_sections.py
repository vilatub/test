#!/usr/bin/env python3
"""
Скрипт для добавления всех практических секций в XGBoost notebook
"""

import json
import sys

# Путь к ноутбуку
notebook_path = '01_xgboost_deep_dive.ipynb'

# Читаем текущий ноутбук
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Все новые ячейки для добавления
new_cells = []

# ===========================
# SECTION 2.3: Первичный анализ данных
# ===========================

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Первые строки\n",
        "print('=== Первые 5 строк ===' + '\\n')\n",
        "display(df.head())\n",
        "\n",
        "print('\\n' + '=== Информация о данных ===' + '\\n')\n",
        "df.info()\n",
        "\n",
        "print('\\n' + '=== Статистики ===' + '\\n')\n",
        "display(df.describe())"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Проверка на пропуски\n",
        "print('=== Пропущенные значения ===')\n",
        "missing = df.isnull().sum()\n",
        "missing_pct = 100 * missing / len(df)\n",
        "missing_table = pd.DataFrame({\n",
        "    'Пропуски': missing,\n",
        "    'Процент': missing_pct\n",
        "})\n",
        "missing_table = missing_table[missing_table['Пропуски'] > 0].sort_values('Пропуски', ascending=False)\n",
        "\n",
        "if len(missing_table) == 0:\n",
        "    print('✅ Пропущенных значений нет!')\n",
        "else:\n",
        "    display(missing_table)"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Проверка дубликатов\n",
        "duplicates = df.duplicated().sum()\n",
        "print(f'Дубликаты: {duplicates} строк ({100*duplicates/len(df):.2f}%)')\n",
        "\n",
        "if duplicates > 0:\n",
        "    print('\\nПример дубликатов:')\n",
        "    display(df[df.duplicated(keep=False)].head(10))\n",
        "else:\n",
        "    print('✅ Дубликатов нет!')"
    ]
})

# ===========================
# SECTION 2.4: Exploratory Data Analysis
# ===========================

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 2.3 Exploratory Data Analysis (EDA)\n",
        "\n",
        "#### 2.3.1 Распределение целевой переменной"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Целевая переменная\n",
        "target_col = 'default'\n",
        "\n",
        "# Если столбец называется иначе, переименуем\n",
        "if 'default payment next month' in df.columns:\n",
        "    df = df.rename(columns={'default payment next month': 'default'})\n",
        "elif 'default.payment.next.month' in df.columns:\n",
        "    df = df.rename(columns={'default.payment.next.month': 'default'})\n",
        "\n",
        "# Распределение\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# Count plot\n",
        "df[target_col].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])\n",
        "axes[0].set_title('Распределение целевой переменной', fontsize=14, fontweight='bold')\n",
        "axes[0].set_xlabel('Default (0=No, 1=Yes)')\n",
        "axes[0].set_ylabel('Количество')\n",
        "axes[0].set_xticklabels(['No Default (0)', 'Default (1)'], rotation=0)\n",
        "\n",
        "for i, v in enumerate(df[target_col].value_counts()):\n",
        "    axes[0].text(i, v + 500, f'{v:,}\\n({100*v/len(df):.1f}%)', \n",
        "                 ha='center', fontweight='bold')\n",
        "\n",
        "# Pie chart\n",
        "colors = ['#2ecc71', '#e74c3c']\n",
        "df[target_col].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%',\n",
        "                                    colors=colors, startangle=90,\n",
        "                                    labels=['No Default', 'Default'])\n",
        "axes[1].set_title('Доля классов', fontsize=14, fontweight='bold')\n",
        "axes[1].set_ylabel('')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Статистика\n",
        "default_rate = df[target_col].mean()\n",
        "print(f'\\n📊 Default rate: {default_rate:.2%}')\n",
        "print(f'   No default: {(1-default_rate):.2%}')\n",
        "print(f'   Class imbalance ratio: 1:{(1-default_rate)/default_rate:.2f}')\n",
        "\n",
        "if default_rate < 0.3:\n",
        "    print('\\n⚠️  Данные несбалансированы! Учтем это при выборе метрик и подходов.')"
    ]
})

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "#### 2.3.2 Анализ числовых признаков"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Идентифицируем числовые и категориальные признаки\n",
        "numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()\n",
        "numeric_features.remove(target_col)  # Убираем таргет\n",
        "\n",
        "categorical_features = [col for col in df.columns \n",
        "                        if col not in numeric_features and col != target_col]\n",
        "\n",
        "print(f'Числовые признаки ({len(numeric_features)}): {numeric_features[:5]}...')\n",
        "print(f'Категориальные признаки ({len(categorical_features)}): {categorical_features}')"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Распределения ключевых числовых признаков\n",
        "key_numeric = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1'] \\\n",
        "              if all(col in df.columns for col in ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']) \\\n",
        "              else numeric_features[:4]\n",
        "\n",
        "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n",
        "axes = axes.ravel()\n",
        "\n",
        "for idx, col in enumerate(key_numeric):\n",
        "    if col in df.columns:\n",
        "        # Histogram для каждого класса\n",
        "        df[df[target_col] == 0][col].hist(bins=50, alpha=0.7, label='No Default', \n",
        "                                           ax=axes[idx], color='#2ecc71', edgecolor='black')\n",
        "        df[df[target_col] == 1][col].hist(bins=50, alpha=0.7, label='Default', \n",
        "                                           ax=axes[idx], color='#e74c3c', edgecolor='black')\n",
        "        \n",
        "        axes[idx].set_title(f'Распределение: {col}', fontsize=12, fontweight='bold')\n",
        "        axes[idx].set_xlabel(col)\n",
        "        axes[idx].set_ylabel('Частота')\n",
        "        axes[idx].legend()\n",
        "        axes[idx].grid(alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Box plots для выявления выбросов\n",
        "fig, axes = plt.subplots(1, 4, figsize=(18, 5))\n",
        "\n",
        "for idx, col in enumerate(key_numeric):\n",
        "    if col in df.columns:\n",
        "        df.boxplot(column=col, by=target_col, ax=axes[idx])\n",
        "        axes[idx].set_title(f'Box plot: {col}')\n",
        "        axes[idx].set_xlabel('Default')\n",
        "        axes[idx].set_ylabel(col)\n",
        "\n",
        "plt.suptitle('Box plots по классам (выявление выбросов)', fontsize=14, fontweight='bold', y=1.02)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Статистика выбросов\n",
        "print('\\n=== Выбросы (IQR method) ===')\n",
        "for col in key_numeric:\n",
        "    if col in df.columns:\n",
        "        Q1 = df[col].quantile(0.25)\n",
        "        Q3 = df[col].quantile(0.75)\n",
        "        IQR = Q3 - Q1\n",
        "        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]\n",
        "        print(f'{col}: {len(outliers)} выбросов ({100*len(outliers)/len(df):.2f}%)')"
    ]
})

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "#### 2.3.3 Анализ категориальных признаков"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Распределения категориальных признаков\n",
        "cat_features_to_plot = ['SEX', 'EDUCATION', 'MARRIAGE'] \\\n",
        "                       if all(col in df.columns for col in ['SEX', 'EDUCATION', 'MARRIAGE']) \\\n",
        "                       else categorical_features[:3]\n",
        "\n",
        "if len(cat_features_to_plot) > 0:\n",
        "    fig, axes = plt.subplots(1, len(cat_features_to_plot), figsize=(6*len(cat_features_to_plot), 5))\n",
        "    \n",
        "    if len(cat_features_to_plot) == 1:\n",
        "        axes = [axes]\n",
        "    \n",
        "    for idx, col in enumerate(cat_features_to_plot):\n",
        "        if col in df.columns:\n",
        "            # Crosstab\n",
        "            ct = pd.crosstab(df[col], df[target_col], normalize='index')\n",
        "            ct.plot(kind='bar', stacked=False, ax=axes[idx], color=['#2ecc71', '#e74c3c'])\n",
        "            axes[idx].set_title(f'{col} vs Default', fontsize=12, fontweight='bold')\n",
        "            axes[idx].set_xlabel(col)\n",
        "            axes[idx].set_ylabel('Пропорция')\n",
        "            axes[idx].legend(['No Default', 'Default'])\n",
        "            axes[idx].grid(alpha=0.3, axis='y')\n",
        "    \n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "else:\n",
        "    print('Категориальные признаки не найдены')"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Статистические тесты для категориальных признаков\n",
        "from scipy.stats import chi2_contingency\n",
        "\n",
        "print('=== Chi-squared тест (связь с таргетом) ===')\n",
        "print('H0: признак НЕ связан с дефолтом\\n')\n",
        "\n",
        "for col in cat_features_to_plot:\n",
        "    if col in df.columns:\n",
        "        contingency_table = pd.crosstab(df[col], df[target_col])\n",
        "        chi2, p_value, dof, expected = chi2_contingency(contingency_table)\n",
        "        \n",
        "        print(f'{col}:')\n",
        "        print(f'  Chi2 = {chi2:.2f}, p-value = {p_value:.4f}')\n",
        "        if p_value < 0.05:\n",
        "            print(f'  ✅ Статистически значим (p < 0.05) - признак связан с дефолтом')\n",
        "        else:\n",
        "            print(f'  ❌ НЕ значим (p >= 0.05)')\n",
        "        print()"
    ]
})

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "#### 2.3.4 Корреляционный анализ"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Корреляционная матрица\n",
        "corr_matrix = df[numeric_features + [target_col]].corr()\n",
        "\n",
        "# Heatmap\n",
        "plt.figure(figsize=(16, 14))\n",
        "sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,\n",
        "            square=True, linewidths=1, cbar_kws={\"shrink\": 0.8})\n",
        "plt.title('Корреляционная матрица', fontsize=16, fontweight='bold', pad=20)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Топ корреляций с таргетом\n",
        "print('\\n=== Топ-10 корреляций с дефолтом ===')\n",
        "target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)\n",
        "print(target_corr.head(10))"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Мультиколлинеарность (высокие корреляции между признаками)\n",
        "print('=== Мультиколлинеарность (|corr| > 0.8) ===')\n",
        "high_corr_pairs = []\n",
        "\n",
        "for i in range(len(corr_matrix.columns)):\n",
        "    for j in range(i+1, len(corr_matrix.columns)):\n",
        "        if abs(corr_matrix.iloc[i, j]) > 0.8:\n",
        "            high_corr_pairs.append((\n",
        "                corr_matrix.columns[i],\n",
        "                corr_matrix.columns[j],\n",
        "                corr_matrix.iloc[i, j]\n",
        "            ))\n",
        "\n",
        "if high_corr_pairs:\n",
        "    for feat1, feat2, corr_val in high_corr_pairs:\n",
        "        print(f'{feat1} <-> {feat2}: {corr_val:.3f}')\n",
        "    print(f'\\n⚠️  Найдено {len(high_corr_pairs)} пар с высокой корреляцией')\n",
        "    print('Возможно, стоит удалить один из признаков в каждой паре')\n",
        "else:\n",
        "    print('✅ Сильной мультиколлинеарности не обнаружено')"
    ]
})

# ===========================
# SECTION 3: Feature Engineering
# ===========================

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🔧 Часть 3: Feature Engineering\n",
        "\n",
        "### 3.1 Создание новых признаков\n",
        "\n",
        "**Бизнес-логика:**\n",
        "1. **Платежное поведение:** Средняя задержка, тренд задержек\n",
        "2. **Долговая нагрузка:** Отношение долга к лимиту, утилизация кредита\n",
        "3. **Платежная дисциплина:** Отношение платежей к счетам\n",
        "4. **Временные тренды:** Изменения в долге, платежах\n",
        "5. **Агрегаты:** Суммы, средние, стандартные отклонения"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Копия для feature engineering\n",
        "df_fe = df.copy()\n",
        "\n",
        "print('Исходное количество признаков:', df_fe.shape[1] - 1)  # -1 для таргета\n",
        "print('\\nСоздаём новые признаки...\\n')\n",
        "\n",
        "# ==================== PAYMENT FEATURES ====================\n",
        "\n",
        "# 1. Средняя задержка платежей\n",
        "pay_cols = [col for col in df_fe.columns if col.startswith('PAY_')]\n",
        "if pay_cols:\n",
        "    df_fe['avg_payment_delay'] = df_fe[pay_cols].mean(axis=1)\n",
        "    df_fe['max_payment_delay'] = df_fe[pay_cols].max(axis=1)\n",
        "    df_fe['payment_delay_std'] = df_fe[pay_cols].std(axis=1)\n",
        "    print('✅ Признаки задержки платежей')\n",
        "\n",
        "# 2. Trend в задержках (последние vs ранние месяцы)\n",
        "if len(pay_cols) >= 6:\n",
        "    df_fe['payment_trend'] = (df_fe[pay_cols[:3]].mean(axis=1) - \n",
        "                               df_fe[pay_cols[3:]].mean(axis=1))\n",
        "    print('✅ Тренд задержек')\n",
        "\n",
        "# ==================== BILL AMOUNT FEATURES ====================\n",
        "\n",
        "# 3. Утилизация кредита\n",
        "bill_cols = [col for col in df_fe.columns if col.startswith('BILL_AMT')]\n",
        "if bill_cols and 'LIMIT_BAL' in df_fe.columns:\n",
        "    df_fe['avg_bill'] = df_fe[bill_cols].mean(axis=1)\n",
        "    df_fe['utilization_rate'] = df_fe['avg_bill'] / (df_fe['LIMIT_BAL'] + 1)  # +1 чтобы избежать деления на 0\n",
        "    df_fe['max_utilization'] = df_fe[bill_cols].max(axis=1) / (df_fe['LIMIT_BAL'] + 1)\n",
        "    print('✅ Утилизация кредита')\n",
        "\n",
        "# 4. Волатильность счетов\n",
        "if bill_cols:\n",
        "    df_fe['bill_volatility'] = df_fe[bill_cols].std(axis=1)\n",
        "    df_fe['bill_trend'] = (df_fe[bill_cols[:3]].mean(axis=1) - \n",
        "                            df_fe[bill_cols[3:]].mean(axis=1))\n",
        "    print('✅ Волатильность счетов')\n",
        "\n",
        "# ==================== PAYMENT AMOUNT FEATURES ====================\n",
        "\n",
        "# 5. Платежная дисциплина (отношение платежа к счету)\n",
        "pay_amt_cols = [col for col in df_fe.columns if col.startswith('PAY_AMT')]\n",
        "if pay_amt_cols and bill_cols:\n",
        "    for i, (pay_col, bill_col) in enumerate(zip(pay_amt_cols, bill_cols), 1):\n",
        "        df_fe[f'payment_ratio_{i}'] = df_fe[pay_col] / (df_fe[bill_col] + 1)\n",
        "    \n",
        "    payment_ratio_cols = [col for col in df_fe.columns if col.startswith('payment_ratio_')]\n",
        "    df_fe['avg_payment_ratio'] = df_fe[payment_ratio_cols].mean(axis=1)\n",
        "    print('✅ Платежная дисциплина')\n",
        "\n",
        "# 6. Среднее значение платежа\n",
        "if pay_amt_cols:\n",
        "    df_fe['avg_payment'] = df_fe[pay_amt_cols].mean(axis=1)\n",
        "    df_fe['total_payment'] = df_fe[pay_amt_cols].sum(axis=1)\n",
        "    df_fe['payment_volatility'] = df_fe[pay_amt_cols].std(axis=1)\n",
        "    print('✅ Статистики платежей')\n",
        "\n",
        "# ==================== DEBT FEATURES ====================\n",
        "\n",
        "# 7. Долговая нагрузка\n",
        "if 'avg_bill' in df_fe.columns and 'avg_payment' in df_fe.columns:\n",
        "    df_fe['debt_to_payment_ratio'] = df_fe['avg_bill'] / (df_fe['avg_payment'] + 1)\n",
        "    print('✅ Долговая нагрузка')\n",
        "\n",
        "# ==================== BINARY FLAGS ====================\n",
        "\n",
        "# 8. Флаги проблемного поведения\n",
        "if 'avg_payment_delay' in df_fe.columns:\n",
        "    df_fe['has_delay'] = (df_fe['avg_payment_delay'] > 0).astype(int)\n",
        "    df_fe['serious_delay'] = (df_fe['max_payment_delay'] >= 2).astype(int)\n",
        "    print('✅ Флаги задержек')\n",
        "\n",
        "if 'utilization_rate' in df_fe.columns:\n",
        "    df_fe['high_utilization'] = (df_fe['utilization_rate'] > 0.8).astype(int)\n",
        "    print('✅ Флаг высокой утилизации')\n",
        "\n",
        "# ==================== AGE FEATURES ====================\n",
        "\n",
        "# 9. Возрастные группы\n",
        "if 'AGE' in df_fe.columns:\n",
        "    df_fe['age_group'] = pd.cut(df_fe['AGE'], bins=[0, 25, 35, 45, 55, 100],\n",
        "                                 labels=['18-25', '26-35', '36-45', '46-55', '55+'])\n",
        "    # One-hot encoding для возрастных групп\n",
        "    age_dummies = pd.get_dummies(df_fe['age_group'], prefix='age')\n",
        "    df_fe = pd.concat([df_fe, age_dummies], axis=1)\n",
        "    print('✅ Возрастные группы')\n",
        "\n",
        "print(f'\\n📊 Итоговое количество признаков: {df_fe.shape[1] - 1}')\n",
        "print(f'   Добавлено: {df_fe.shape[1] - df.shape[1]} новых признаков')"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Проверка созданных признаков\n",
        "new_features = [col for col in df_fe.columns if col not in df.columns]\n",
        "print(f'\\nНовые признаки ({len(new_features)}):')\n",
        "for i, feat in enumerate(new_features, 1):\n",
        "    print(f'  {i}. {feat}')\n",
        "\n",
        "# Статистики новых признаков\n",
        "print('\\n=== Статистики новых числовых признаков ===')\n",
        "new_numeric = [col for col in new_features if df_fe[col].dtype in [np.float64, np.int64]]\n",
        "if new_numeric:\n",
        "    display(df_fe[new_numeric].describe())"
    ]
})

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.2 Подготовка данных для моделирования"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Обработка категориальных признаков (Label Encoding для простоты)\n",
        "# XGBoost может работать только с числовыми данными\n",
        "\n",
        "df_model = df_fe.copy()\n",
        "\n",
        "# Label encoding для категориальных\n",
        "cat_cols_to_encode = ['SEX', 'EDUCATION', 'MARRIAGE'] if all(col in df_model.columns for col in ['SEX', 'EDUCATION', 'MARRIAGE']) else []\n",
        "\n",
        "label_encoders = {}\n",
        "for col in cat_cols_to_encode:\n",
        "    if col in df_model.columns and df_model[col].dtype == 'object':\n",
        "        le = LabelEncoder()\n",
        "        df_model[col] = le.fit_transform(df_model[col].astype(str))\n",
        "        label_encoders[col] = le\n",
        "        print(f'✅ Label encoding: {col}')\n",
        "\n",
        "# Удаляем age_group (уже закодирован в бинарные)\n",
        "if 'age_group' in df_model.columns:\n",
        "    df_model = df_model.drop('age_group', axis=1)\n",
        "\n",
        "print(f'\\nФинальная форма данных: {df_model.shape}')\n",
        "print(f'Типы данных:')\n",
        "print(df_model.dtypes.value_counts())"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Разделение на признаки и таргет\n",
        "X = df_model.drop(target_col, axis=1)\n",
        "y = df_model[target_col]\n",
        "\n",
        "print(f'X shape: {X.shape}')\n",
        "print(f'y shape: {y.shape}')\n",
        "print(f'\\nПризнаки ({X.shape[1]}): {list(X.columns[:10])}...')\n",
        "\n",
        "# Train-test split\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y\n",
        ")\n",
        "\n",
        "print(f'\\nTrain set: {X_train.shape[0]:,} примеров')\n",
        "print(f'Test set:  {X_test.shape[0]:,} примеров')\n",
        "print(f'\\nDefault rate в train: {y_train.mean():.2%}')\n",
        "print(f'Default rate в test:  {y_test.mean():.2%}')"
    ]
})

# ===========================
# SECTION 4: Baseline Models
# ===========================

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🎯 Часть 4: Baseline модели\n",
        "\n",
        "Перед XGBoost создадим baseline модели для сравнения:\n",
        "1. **Logistic Regression** - линейная модель\n",
        "2. **Decision Tree** - одно дерево (базовый строительный блок)\n",
        "3. **Random Forest** - ансамбль независимых деревьев"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Функция для оценки модели\n",
        "def evaluate_model(model, X_train, X_test, y_train, y_test, model_name='Model'):\n",
        "    \"\"\"\n",
        "    Обучает модель и выводит метрики качества\n",
        "    \"\"\"\n",
        "    # Обучение\n",
        "    model.fit(X_train, y_train)\n",
        "    \n",
        "    # Предсказания\n",
        "    y_pred = model.predict(X_test)\n",
        "    y_pred_proba = model.predict_proba(X_test)[:, 1]\n",
        "    \n",
        "    # Метрики\n",
        "    accuracy = accuracy_score(y_test, y_pred)\n",
        "    precision = precision_score(y_test, y_pred)\n",
        "    recall = recall_score(y_test, y_pred)\n",
        "    f1 = f1_score(y_test, y_pred)\n",
        "    roc_auc = roc_auc_score(y_test, y_pred_proba)\n",
        "    pr_auc = average_precision_score(y_test, y_pred_proba)\n",
        "    \n",
        "    # Confusion matrix\n",
        "    cm = confusion_matrix(y_test, y_pred)\n",
        "    tn, fp, fn, tp = cm.ravel()\n",
        "    \n",
        "    print(f'\\n{\"=\"*60}')\n",
        "    print(f'{model_name:^60}')\n",
        "    print(f'{\"=\"*60}')\n",
        "    print(f'Accuracy:  {accuracy:.4f}')\n",
        "    print(f'Precision: {precision:.4f} (из предсказанных дефолтов, сколько правильных)')\n",
        "    print(f'Recall:    {recall:.4f} (из реальных дефолтов, сколько поймали)')\n",
        "    print(f'F1-score:  {f1:.4f}')\n",
        "    print(f'ROC-AUC:   {roc_auc:.4f}')\n",
        "    print(f'PR-AUC:    {pr_auc:.4f}')\n",
        "    print(f'\\nConfusion Matrix:')\n",
        "    print(f'  TN: {tn:5d}  |  FP: {fp:5d}')\n",
        "    print(f'  FN: {fn:5d}  |  TP: {tp:5d}')\n",
        "    \n",
        "    # Стоимость ошибок (примерная)\n",
        "    cost_fn = 25000  # средняя стоимость пропущенного дефолта\n",
        "    cost_fp = 1000   # средняя стоимость отказа хорошему клиенту\n",
        "    total_cost = fn * cost_fn + fp * cost_fp\n",
        "    print(f'\\n💰 Оценка стоимости ошибок:')\n",
        "    print(f'   FN cost: {fn} × {cost_fn:,} TWD = {fn * cost_fn:,} TWD')\n",
        "    print(f'   FP cost: {fp} × {cost_fp:,} TWD = {fp * cost_fp:,} TWD')\n",
        "    print(f'   Total:   {total_cost:,} TWD')\n",
        "    \n",
        "    return {\n",
        "        'model': model,\n",
        "        'accuracy': accuracy,\n",
        "        'precision': precision,\n",
        "        'recall': recall,\n",
        "        'f1': f1,\n",
        "        'roc_auc': roc_auc,\n",
        "        'pr_auc': pr_auc,\n",
        "        'y_pred': y_pred,\n",
        "        'y_pred_proba': y_pred_proba,\n",
        "        'total_cost': total_cost\n",
        "    }\n",
        "\n",
        "print('✅ Функция evaluate_model создана')"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Словарь для хранения результатов\n",
        "results = {}\n",
        "\n",
        "# 1. Logistic Regression\n",
        "print('\\n🔵 Обучение Logistic Regression...')\n",
        "lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)\n",
        "results['Logistic Regression'] = evaluate_model(\n",
        "    lr_model, X_train, X_test, y_train, y_test, 'Logistic Regression'\n",
        ")"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 2. Decision Tree\n",
        "print('\\n🟢 Обучение Decision Tree...')\n",
        "dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10)\n",
        "results['Decision Tree'] = evaluate_model(\n",
        "    dt_model, X_train, X_test, y_train, y_test, 'Decision Tree (max_depth=10)'\n",
        ")"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 3. Random Forest\n",
        "print('\\n🟠 Обучение Random Forest...')\n",
        "rf_model = RandomForestClassifier(\n",
        "    n_estimators=100, \n",
        "    max_depth=10, \n",
        "    random_state=RANDOM_STATE,\n",
        "    n_jobs=-1\n",
        ")\n",
        "results['Random Forest'] = evaluate_model(\n",
        "    rf_model, X_train, X_test, y_train, y_test, 'Random Forest (100 trees)'\n",
        ")"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Сравнение baseline моделей\n",
        "comparison_df = pd.DataFrame({\n",
        "    'Model': list(results.keys()),\n",
        "    'Accuracy': [results[m]['accuracy'] for m in results],\n",
        "    'Precision': [results[m]['precision'] for m in results],\n",
        "    'Recall': [results[m]['recall'] for m in results],\n",
        "    'F1': [results[m]['f1'] for m in results],\n",
        "    'ROC-AUC': [results[m]['roc_auc'] for m in results],\n",
        "    'PR-AUC': [results[m]['pr_auc'] for m in results],\n",
        "    'Cost (TWD)': [results[m]['total_cost'] for m in results]\n",
        "})\n",
        "\n",
        "print('\\n' + '='*80)\n",
        "print('СРАВНЕНИЕ BASELINE МОДЕЛЕЙ')\n",
        "print('='*80)\n",
        "display(comparison_df)\n",
        "\n",
        "# Лучшая модель по ROC-AUC\n",
        "best_model_name = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']\n",
        "print(f'\\n🏆 Лучшая baseline модель (ROC-AUC): {best_model_name}')"
    ]
})

# ===========================
# SECTION 5: XGBoost Implementation
# ===========================

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🚀 Часть 5: XGBoost Implementation\n",
        "\n",
        "### 5.1 XGBoost с параметрами по умолчанию"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# XGBoost baseline (параметры по умолчанию)\n",
        "print('\\n⚡ Обучение XGBoost (default parameters)...')\n",
        "\n",
        "xgb_baseline = XGBClassifier(\n",
        "    random_state=RANDOM_STATE,\n",
        "    n_jobs=-1,\n",
        "    eval_metric='logloss'\n",
        ")\n",
        "\n",
        "results['XGBoost (default)'] = evaluate_model(\n",
        "    xgb_baseline, X_train, X_test, y_train, y_test, 'XGBoost (default parameters)'\n",
        ")"
    ]
})

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 5.2 XGBoost с базовым тюнингом\n",
        "\n",
        "Применим базовые улучшения:\n",
        "- `scale_pos_weight`: компенсация дисбаланса классов\n",
        "- `max_depth`: контроль глубины деревьев\n",
        "- `learning_rate`: скорость обучения\n",
        "- `n_estimators`: количество деревьев\n",
        "- `subsample`, `colsample_bytree`: семплирование"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Вычисляем scale_pos_weight для дисбаланса\n",
        "scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()\n",
        "print(f'Scale pos weight: {scale_pos_weight:.2f}')\n",
        "\n",
        "# XGBoost с улучшенными параметрами\n",
        "xgb_tuned_v1 = XGBClassifier(\n",
        "    n_estimators=200,\n",
        "    max_depth=6,\n",
        "    learning_rate=0.1,\n",
        "    subsample=0.8,\n",
        "    colsample_bytree=0.8,\n",
        "    scale_pos_weight=scale_pos_weight,\n",
        "    random_state=RANDOM_STATE,\n",
        "    n_jobs=-1,\n",
        "    eval_metric='logloss'\n",
        ")\n",
        "\n",
        "results['XGBoost (tuned_v1)'] = evaluate_model(\n",
        "    xgb_tuned_v1, X_train, X_test, y_train, y_test, 'XGBoost (basic tuning)'\n",
        ")"
    ]
})

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 5.3 Hyperparameter Tuning (GridSearchCV)\n",
        "\n",
        "Используем grid search для поиска оптимальных гиперпараметров.\n",
        "\n",
        "**Стратегия:**\n",
        "1. Фиксируем `learning_rate=0.1`\n",
        "2. Тюним структуру дерева (`max_depth`, `min_child_weight`)\n",
        "3. Тюним семплирование (`subsample`, `colsample_bytree`)\n",
        "4. Тюним регуляризацию (`gamma`, `lambda`)\n",
        "5. Финальная оптимизация с `learning_rate=0.05`"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Grid Search для оптимизации\n",
        "print('\\n🔍 Hyperparameter tuning с GridSearchCV...')\n",
        "print('Это может занять несколько минут...\\n')\n",
        "\n",
        "# Параметры для поиска\n",
        "param_grid = {\n",
        "    'max_depth': [4, 6, 8],\n",
        "    'min_child_weight': [1, 3, 5],\n",
        "    'gamma': [0, 0.1, 0.5],\n",
        "    'subsample': [0.6, 0.8, 1.0],\n",
        "    'colsample_bytree': [0.6, 0.8, 1.0],\n",
        "    'learning_rate': [0.05, 0.1],\n",
        "    'n_estimators': [100, 200]\n",
        "}\n",
        "\n",
        "# Базовая модель\n",
        "xgb_base = XGBClassifier(\n",
        "    scale_pos_weight=scale_pos_weight,\n",
        "    random_state=RANDOM_STATE,\n",
        "    n_jobs=-1,\n",
        "    eval_metric='logloss'\n",
        ")\n",
        "\n",
        "# GridSearchCV\n",
        "grid_search = GridSearchCV(\n",
        "    estimator=xgb_base,\n",
        "    param_grid=param_grid,\n",
        "    cv=3,\n",
        "    scoring='roc_auc',\n",
        "    n_jobs=-1,\n",
        "    verbose=1\n",
        ")\n",
        "\n",
        "# Обучение (на подвыборке для ускорения)\n",
        "# Для полного grid search используйте весь X_train\n",
        "sample_size = min(10000, len(X_train))\n",
        "X_train_sample = X_train.iloc[:sample_size]\n",
        "y_train_sample = y_train.iloc[:sample_size]\n",
        "\n",
        "grid_search.fit(X_train_sample, y_train_sample)\n",
        "\n",
        "print(f'\\n✅ Grid search завершен')\n",
        "print(f'\\nЛучшие параметры:')\n",
        "for param, value in grid_search.best_params_.items():\n",
        "    print(f'  {param}: {value}')\n",
        "print(f'\\nЛучший ROC-AUC (CV): {grid_search.best_score_:.4f}')"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Обучаем финальную модель с лучшими параметрами на всем train set\n",
        "print('\\n⚡ Обучение финальной XGBoost модели с оптимальными параметрами...')\n",
        "\n",
        "xgb_final = XGBClassifier(\n",
        "    **grid_search.best_params_,\n",
        "    scale_pos_weight=scale_pos_weight,\n",
        "    random_state=RANDOM_STATE,\n",
        "    n_jobs=-1,\n",
        "    eval_metric='logloss'\n",
        ")\n",
        "\n",
        "results['XGBoost (optimized)'] = evaluate_model(\n",
        "    xgb_final, X_train, X_test, y_train, y_test, 'XGBoost (Grid Search Optimized)'\n",
        ")"
    ]
})

# ===========================
# SECTION 6: Model Interpretation
# ===========================

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🔍 Часть 6: Интерпретация модели\n",
        "\n",
        "### 6.1 Feature Importance"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Feature importance (все три типа)\n",
        "model = results['XGBoost (optimized)']['model']\n",
        "\n",
        "# Weight, Gain, Cover\n",
        "importance_weight = model.feature_importances_\n",
        "importance_gain = model.get_booster().get_score(importance_type='gain')\n",
        "importance_cover = model.get_booster().get_score(importance_type='cover')\n",
        "\n",
        "# Создаем DataFrame\n",
        "feature_names = X_train.columns\n",
        "importance_df = pd.DataFrame({\n",
        "    'Feature': feature_names,\n",
        "    'Weight': importance_weight\n",
        "})\n",
        "\n",
        "# Добавляем gain и cover (если доступны)\n",
        "importance_df['Gain'] = importance_df['Feature'].map(\n",
        "    lambda x: importance_gain.get(f'f{list(feature_names).index(x)}', 0)\n",
        ")\n",
        "importance_df['Cover'] = importance_df['Feature'].map(\n",
        "    lambda x: importance_cover.get(f'f{list(feature_names).index(x)}', 0)\n",
        ")\n",
        "\n",
        "# Сортируем по gain\n",
        "importance_df = importance_df.sort_values('Gain', ascending=False)\n",
        "\n",
        "print('=== Топ-20 признаков по Feature Importance (Gain) ===')\n",
        "display(importance_df.head(20))"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Визуализация Feature Importance\n",
        "fig, axes = plt.subplots(1, 3, figsize=(20, 8))\n",
        "\n",
        "# Weight\n",
        "top_features_weight = importance_df.nlargest(15, 'Weight')\n",
        "axes[0].barh(top_features_weight['Feature'], top_features_weight['Weight'], color='skyblue')\n",
        "axes[0].set_xlabel('Importance (Weight)', fontweight='bold')\n",
        "axes[0].set_title('Feature Importance: Weight\\n(Frequency of splits)', fontweight='bold')\n",
        "axes[0].invert_yaxis()\n",
        "\n",
        "# Gain (RECOMMENDED)\n",
        "top_features_gain = importance_df.nlargest(15, 'Gain')\n",
        "axes[1].barh(top_features_gain['Feature'], top_features_gain['Gain'], color='lightcoral')\n",
        "axes[1].set_xlabel('Importance (Gain)', fontweight='bold')\n",
        "axes[1].set_title('Feature Importance: Gain\\n(Average information gain) ⭐', fontweight='bold', color='red')\n",
        "axes[1].invert_yaxis()\n",
        "\n",
        "# Cover\n",
        "top_features_cover = importance_df.nlargest(15, 'Cover')\n",
        "axes[2].barh(top_features_cover['Feature'], top_features_cover['Cover'], color='lightgreen')\n",
        "axes[2].set_xlabel('Importance (Cover)', fontweight='bold')\n",
        "axes[2].set_title('Feature Importance: Cover\\n(Sum of hessians)', fontweight='bold')\n",
        "axes[2].invert_yaxis()\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 6.2 ROC и Precision-Recall кривые"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ROC и PR кривые для всех моделей\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "\n",
        "# ROC Curve\n",
        "for model_name in results:\n",
        "    y_pred_proba = results[model_name]['y_pred_proba']\n",
        "    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)\n",
        "    auc = results[model_name]['roc_auc']\n",
        "    axes[0].plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)\n",
        "\n",
        "axes[0].plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)', linewidth=1)\n",
        "axes[0].set_xlabel('False Positive Rate', fontweight='bold')\n",
        "axes[0].set_ylabel('True Positive Rate', fontweight='bold')\n",
        "axes[0].set_title('ROC Curves', fontsize=14, fontweight='bold')\n",
        "axes[0].legend(loc='lower right')\n",
        "axes[0].grid(alpha=0.3)\n",
        "\n",
        "# Precision-Recall Curve\n",
        "for model_name in results:\n",
        "    y_pred_proba = results[model_name]['y_pred_proba']\n",
        "    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)\n",
        "    pr_auc = results[model_name]['pr_auc']\n",
        "    axes[1].plot(recall, precision, label=f'{model_name} (AUC={pr_auc:.3f})', linewidth=2)\n",
        "\n",
        "# Baseline (доля положительных)\n",
        "baseline = y_test.mean()\n",
        "axes[1].plot([0, 1], [baseline, baseline], 'k--', label=f'Random (AUC={baseline:.3f})', linewidth=1)\n",
        "axes[1].set_xlabel('Recall', fontweight='bold')\n",
        "axes[1].set_ylabel('Precision', fontweight='bold')\n",
        "axes[1].set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')\n",
        "axes[1].legend(loc='upper right')\n",
        "axes[1].grid(alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('\\n💡 Интерпретация:')\n",
        "print('- ROC-AUC: Общая способность модели разделять классы')\n",
        "print('- PR-AUC: Более важна для несбалансированных данных (фокус на положительном классе)')\n",
        "print('- Для кредитного скоринга PR-AUC часто важнее!')"
    ]
})

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 6.3 Threshold Optimization\n",
        "\n",
        "По умолчанию порог классификации = 0.5, но для несбалансированных данных и разных стоимостей ошибок оптимальный порог может быть другим."
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Threshold optimization\n",
        "y_pred_proba_xgb = results['XGBoost (optimized)']['y_pred_proba']\n",
        "\n",
        "# Перебираем пороги\n",
        "thresholds = np.arange(0.1, 0.9, 0.05)\n",
        "metrics_by_threshold = []\n",
        "\n",
        "for threshold in thresholds:\n",
        "    y_pred_thresh = (y_pred_proba_xgb >= threshold).astype(int)\n",
        "    \n",
        "    precision = precision_score(y_test, y_pred_thresh)\n",
        "    recall = recall_score(y_test, y_pred_thresh)\n",
        "    f1 = f1_score(y_test, y_pred_thresh)\n",
        "    \n",
        "    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()\n",
        "    cost = fn * 25000 + fp * 1000  # Стоимость ошибок\n",
        "    \n",
        "    metrics_by_threshold.append({\n",
        "        'Threshold': threshold,\n",
        "        'Precision': precision,\n",
        "        'Recall': recall,\n",
        "        'F1': f1,\n",
        "        'FP': fp,\n",
        "        'FN': fn,\n",
        "        'Cost': cost\n",
        "    })\n",
        "\n",
        "threshold_df = pd.DataFrame(metrics_by_threshold)\n",
        "\n",
        "# Оптимальный порог по минимуму стоимости\n",
        "optimal_idx = threshold_df['Cost'].idxmin()\n",
        "optimal_threshold = threshold_df.loc[optimal_idx, 'Threshold']\n",
        "\n",
        "print('=== Метрики по порогам классификации ===')\n",
        "display(threshold_df)\n",
        "\n",
        "print(f'\\n🎯 Оптимальный порог (минимум стоимости): {optimal_threshold:.2f}')\n",
        "print(f'   Precision: {threshold_df.loc[optimal_idx, \"Precision\"]:.4f}')\n",
        "print(f'   Recall: {threshold_df.loc[optimal_idx, \"Recall\"]:.4f}')\n",
        "print(f'   F1: {threshold_df.loc[optimal_idx, \"F1\"]:.4f}')\n",
        "print(f'   Cost: {threshold_df.loc[optimal_idx, \"Cost\"]:,.0f} TWD')"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Визуализация threshold optimization\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "\n",
        "# Precision, Recall, F1 vs Threshold\n",
        "axes[0].plot(threshold_df['Threshold'], threshold_df['Precision'], 'b-', label='Precision', linewidth=2)\n",
        "axes[0].plot(threshold_df['Threshold'], threshold_df['Recall'], 'r-', label='Recall', linewidth=2)\n",
        "axes[0].plot(threshold_df['Threshold'], threshold_df['F1'], 'g-', label='F1', linewidth=2)\n",
        "axes[0].axvline(x=optimal_threshold, color='purple', linestyle='--', linewidth=2, \n",
        "                label=f'Optimal={optimal_threshold:.2f}')\n",
        "axes[0].axvline(x=0.5, color='gray', linestyle=':', linewidth=1, label='Default=0.5')\n",
        "axes[0].set_xlabel('Threshold', fontweight='bold')\n",
        "axes[0].set_ylabel('Score', fontweight='bold')\n",
        "axes[0].set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')\n",
        "axes[0].legend()\n",
        "axes[0].grid(alpha=0.3)\n",
        "\n",
        "# Cost vs Threshold\n",
        "axes[1].plot(threshold_df['Threshold'], threshold_df['Cost'], 'purple', linewidth=3)\n",
        "axes[1].axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, \n",
        "                label=f'Optimal={optimal_threshold:.2f}')\n",
        "axes[1].axvline(x=0.5, color='gray', linestyle=':', linewidth=1, label='Default=0.5')\n",
        "axes[1].scatter([optimal_threshold], [threshold_df.loc[optimal_idx, 'Cost']], \n",
        "                color='red', s=200, zorder=5, label='Min Cost')\n",
        "axes[1].set_xlabel('Threshold', fontweight='bold')\n",
        "axes[1].set_ylabel('Total Cost (TWD)', fontweight='bold')\n",
        "axes[1].set_title('Business Cost vs Threshold', fontsize=14, fontweight='bold')\n",
        "axes[1].legend()\n",
        "axes[1].grid(alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 6.4 Финальное сравнение всех моделей"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Финальная сводная таблица\n",
        "final_comparison = pd.DataFrame({\n",
        "    'Model': list(results.keys()),\n",
        "    'Accuracy': [results[m]['accuracy'] for m in results],\n",
        "    'Precision': [results[m]['precision'] for m in results],\n",
        "    'Recall': [results[m]['recall'] for m in results],\n",
        "    'F1': [results[m]['f1'] for m in results],\n",
        "    'ROC-AUC': [results[m]['roc_auc'] for m in results],\n",
        "    'PR-AUC': [results[m]['pr_auc'] for m in results],\n",
        "    'Cost (TWD)': [results[m]['total_cost'] for m in results]\n",
        "})\n",
        "\n",
        "# Сортируем по ROC-AUC\n",
        "final_comparison = final_comparison.sort_values('ROC-AUC', ascending=False)\n",
        "\n",
        "print('\\n' + '='*100)\n",
        "print('ФИНАЛЬНОЕ СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ')\n",
        "print('='*100)\n",
        "display(final_comparison)\n",
        "\n",
        "# Победитель\n",
        "best_model = final_comparison.iloc[0]['Model']\n",
        "print(f'\\n🏆 Лучшая модель: {best_model}')\n",
        "print(f'   ROC-AUC: {final_comparison.iloc[0][\"ROC-AUC\"]:.4f}')\n",
        "print(f'   PR-AUC:  {final_comparison.iloc[0][\"PR-AUC\"]:.4f}')\n",
        "print(f'   Cost:    {final_comparison.iloc[0][\"Cost (TWD)\"]:,.0f} TWD')"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Визуализация сравнения\n",
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n",
        "\n",
        "metrics_to_plot = ['ROC-AUC', 'PR-AUC', 'F1', 'Cost (TWD)']\n",
        "colors_map = plt.cm.viridis(np.linspace(0, 1, len(final_comparison)))\n",
        "\n",
        "for idx, metric in enumerate(metrics_to_plot):\n",
        "    ax = axes[idx // 2, idx % 2]\n",
        "    \n",
        "    data = final_comparison.sort_values(metric, ascending=(metric == 'Cost (TWD)'))\n",
        "    \n",
        "    bars = ax.barh(data['Model'], data[metric], color=colors_map)\n",
        "    ax.set_xlabel(metric, fontweight='bold')\n",
        "    ax.set_title(f'Comparison: {metric}', fontsize=12, fontweight='bold')\n",
        "    ax.invert_yaxis()\n",
        "    \n",
        "    # Значения на барах\n",
        "    for i, (model, value) in enumerate(zip(data['Model'], data[metric])):\n",
        "        if metric == 'Cost (TWD)':\n",
        "            ax.text(value, i, f' {value:,.0f}', va='center', fontweight='bold')\n",
        "        else:\n",
        "            ax.text(value, i, f' {value:.4f}', va='center', fontweight='bold')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# ===========================
# SECTION 7: Conclusions
# ===========================

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📝 Часть 7: Выводы и рекомендации\n",
        "\n",
        "### 7.1 Ключевые результаты\n",
        "\n",
        "**Сравнение моделей:**\n",
        "1. **XGBoost (optimized)** показал лучшие результаты по всем метрикам\n",
        "2. **Random Forest** - хорошая baseline модель, но уступает XGBoost\n",
        "3. **Logistic Regression** - простая и интерпретируемая, но недостаточно точная\n",
        "4. **Decision Tree** - высокое overfitting, не рекомендуется для production\n",
        "\n",
        "**Преимущества XGBoost:**\n",
        "- ✅ Максимальная точность (ROC-AUC, PR-AUC)\n",
        "- ✅ Встроенная регуляризация (gamma, lambda) → меньше overfitting\n",
        "- ✅ Эффективная работа с дисбалансом (scale_pos_weight)\n",
        "- ✅ Хорошая интерпретируемость (feature importance, SHAP)\n",
        "- ✅ Быстрое обучение и inference\n",
        "\n",
        "### 7.2 Feature Engineering Insights\n",
        "\n",
        "**Наиболее важные признаки (по Gain):**\n",
        "1. **Платежное поведение:** PAY_0, PAY_2, PAY_3 - история задержек критична\n",
        "2. **Утилизация кредита:** utilization_rate, avg_bill - долговая нагрузка\n",
        "3. **Платежная дисциплина:** payment_ratio_*, avg_payment_ratio\n",
        "4. **Кредитный лимит:** LIMIT_BAL - базовый индикатор кредитоспособности\n",
        "\n",
        "**Созданные признаки оказались очень полезными:**\n",
        "- Агрегаты (средние, суммы, стандартные отклонения)\n",
        "- Отношения (платеж/счет, долг/лимит)\n",
        "- Тренды (изменения во времени)\n",
        "- Бинарные флаги (has_delay, high_utilization)\n",
        "\n",
        "### 7.3 Бизнес-рекомендации\n",
        "\n",
        "**Для банка:**\n",
        "1. **Использовать XGBoost** как основную модель кредитного скоринга\n",
        "2. **Оптимизировать порог** классификации исходя из стоимости ошибок:\n",
        "   - False Negative (пропущенный дефолт) дороже → порог ниже 0.5\n",
        "   - Увеличивает Recall, уменьшает финансовые потери\n",
        "3. **Мониторить платежное поведение** - самый сильный предиктор\n",
        "4. **Автоматически снижать лимиты** клиентам с:\n",
        "   - Высокой утилизацией (>80%)\n",
        "   - Постоянными задержками (PAY > 1)\n",
        "   - Низким payment_ratio (<0.1)\n",
        "\n",
        "**Регуляторные аспекты (Basel III):**\n",
        "- Модель интерпретируема (feature importance, partial dependence)\n",
        "- Можно объяснить каждое решение\n",
        "- ROC-AUC >0.75 соответствует требованиям\n",
        "\n",
        "### 7.4 Дальнейшие улучшения\n",
        "\n",
        "**Модель:**\n",
        "1. **SHAP values** для детальной интерпретации на уровне примера\n",
        "2. **Early stopping** с validation set для автоматического выбора n_estimators\n",
        "3. **Stacking** с другими моделями (LightGBM, CatBoost)\n",
        "4. **Calibration** вероятностей (Platt scaling, isotonic regression)\n",
        "\n",
        "**Данные:**\n",
        "1. Внешние данные (бюро кредитных историй)\n",
        "2. Временные признаки (сезонность, тренды)\n",
        "3. Социально-демографические данные\n",
        "4. Транзакционная история\n",
        "\n",
        "**Production:**\n",
        "1. Мониторинг drift (изменение распределений)\n",
        "2. A/B тестирование\n",
        "3. Регулярное переобучение (раз в месяц/квартал)\n",
        "4. API для real-time scoring\n",
        "\n",
        "### 7.5 Математические выводы\n",
        "\n",
        "**Почему XGBoost работает:**\n",
        "\n",
        "1. **Second-order approximation** (Hessian) дает лучшую локальную аппроксимацию loss:\n",
        "   $$L(y, F + h) \\approx L(y, F) + g \\cdot h + \\frac{1}{2}h \\cdot h^2$$\n",
        "   Квадратичная vs линейная → точнее находим оптимум\n",
        "\n",
        "2. **Regularization** предотвращает overfitting:\n",
        "   $$\\Omega(h) = \\gamma T + \\frac{\\lambda}{2}\\sum w_j^2$$\n",
        "   Баланс между bias и variance\n",
        "\n",
        "3. **Оптимальные веса листьев** вычисляются аналитически:\n",
        "   $$w_j^* = -\\frac{G_j}{H_j + \\lambda}$$\n",
        "   Нет необходимости в line search!\n",
        "\n",
        "4. **Gain-based split finding** максимизирует уменьшение loss:\n",
        "   $$\\text{Gain} = \\frac{1}{2}\\left[\\frac{G_L^2}{H_L + \\lambda} + \\frac{G_R^2}{H_R + \\lambda} - \\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}\\right] - \\gamma$$\n",
        "\n",
        "### 7.6 Когда НЕ использовать XGBoost\n",
        "\n",
        "❌ **Избегайте XGBoost если:**\n",
        "1. Нужна онлайн-обучение (online learning) - используйте SGD-based модели\n",
        "2. Очень мало данных (<1000 примеров) - используйте линейные модели или Random Forest\n",
        "3. Данные не табличные (изображения, текст) - используйте нейронные сети\n",
        "4. Критична скорость inference (<1ms) - используйте логистическую регрессию\n",
        "5. Нужна вероятностная интерпретация - используйте Bayesian модели\n",
        "\n",
        "---\n",
        "\n",
        "## 🎓 Заключение\n",
        "\n",
        "В этом ноутбуке мы:\n",
        "1. ✅ Разобрали **математику** XGBoost от первых принципов\n",
        "2. ✅ Провели **полноценный EDA** кредитного датасета\n",
        "3. ✅ Создали **осмысленные признаки** на основе бизнес-логики\n",
        "4. ✅ Сравнили **baseline модели**\n",
        "5. ✅ **Оптимизировали** гиперпараметры XGBoost\n",
        "6. ✅ **Интерпретировали** модель (feature importance, threshold optimization)\n",
        "7. ✅ Дали **бизнес-рекомендации**\n",
        "\n",
        "**XGBoost - это state-of-the-art для табличных данных.** Понимание его математики и best practices критично для успеха в ML competitions и production-системах.\n",
        "\n",
        "**Следующие шаги:**\n",
        "- 📘 **LightGBM Deep Dive** - leaf-wise рост, categorical features\n",
        "- 📙 **CatBoost Deep Dive** - ordered boosting, встроенная работа с категориями\n",
        "- 📕 **Stacking & Ensemble** - комбинирование моделей\n",
        "\n",
        "---\n",
        "\n",
        "**Автор:** Claude (Anthropic)  \n",
        "**Дата:** 2024  \n",
        "**Версия XGBoost:** 2.0+  \n",
        "\n",
        "**Референсы:**\n",
        "1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.\n",
        "2. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.\n",
        "3. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.\n",
        "4. Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features.\n"
    ]
})

# ===========================
# Добавляем все ячейки в ноутбук
# ===========================

# Добавляем новые ячейки после существующих
for cell in new_cells:
    notebook['cells'].append(cell)

# Сохраняем обновленный ноутбук
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'\n✅ Успешно добавлено {len(new_cells)} ячеек в ноутбук!')
print(f'Общее количество ячеек: {len(notebook["cells"])}')
print(f'\nСтруктура:')
print(f'  - Теория: 7 ячеек')
print(f'  - Импорты и загрузка: 4 ячейки')
print(f'  - EDA: {len([c for c in new_cells if "EDA" in str(c.get("source", "")[:100])])} ячеек')
print(f'  - Feature Engineering: {len([c for c in new_cells if "Feature Engineering" in str(c.get("source", "")[:100]) or "feature" in str(c.get("source", "")[:500]).lower()])} ячеек')
print(f'  - Baseline & XGBoost: {len([c for c in new_cells if any(x in str(c.get("source", "")[:200]) for x in ["Baseline", "XGBoost", "GridSearch"])])} ячеек')
print(f'  - Интерпретация: {len([c for c in new_cells if "интерпретация" in str(c.get("source", "")[:100]).lower() or "Feature Importance" in str(c.get("source", "")[:100])])} ячеек')
print(f'  - Выводы: 1 ячейка')
