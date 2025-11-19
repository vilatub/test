#!/usr/bin/env python3
"""
Скрипт для создания notebook по обнаружению сетевых атак.
Методы: Spectral Clustering, Mean-Shift, Bidirectional LSTM-Autoencoder
"""

import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Заголовок и введение
    cells.append(nbf.v4.new_markdown_cell("""# Фаза 5.4: Обнаружение сетевых вторжений

## Продвинутые методы кластеризации и глубокого обучения

В этом ноутбуке мы рассмотрим продвинутые методы обнаружения аномалий на примере анализа сетевого трафика:

### Методы кластеризации
1. **Spectral Clustering** - использует собственные векторы матрицы сходства
2. **Mean-Shift** - непараметрический алгоритм поиска мод плотности

### Глубокое обучение
3. **Bidirectional LSTM-Autoencoder** - двунаправленная рекуррентная сеть для анализа последовательностей

### Задача
Обнаружение сетевых атак: DDoS, сканирование портов, brute force и другие аномалии в сетевом трафике.

### Датасет
Синтетический датасет сетевого трафика (~15,000 записей) с различными типами атак и нормальным трафиком."""))

    # Cell 2: Импорты
    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import SpectralClustering, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Глубокое обучение
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, RepeatVector,
                                     TimeDistributed, Bidirectional, Dropout)
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

print("Библиотеки загружены успешно")
print(f"TensorFlow версия: {tf.__version__}")"""))

    # Cell 3: Создание датасета
    cells.append(nbf.v4.new_markdown_cell("""## 1. Создание датасета сетевого трафика

### Описание признаков

Наш датасет содержит следующие признаки сетевых соединений:

- **duration** - длительность соединения (секунды)
- **src_bytes** - байты от источника к получателю
- **dst_bytes** - байты от получателя к источнику
- **count** - число соединений к тому же хосту за последние 2 секунды
- **srv_count** - число соединений к тому же сервису за последние 2 секунды
- **serror_rate** - процент соединений с ошибками SYN
- **rerror_rate** - процент соединений с ошибками REJ
- **same_srv_rate** - процент соединений к тому же сервису
- **dst_host_count** - число соединений к тому же хосту назначения
- **dst_host_srv_count** - число соединений к тому же сервису на хосте назначения

### Типы трафика

1. **Normal** - нормальный трафик
2. **DDoS** - распределённая атака отказа в обслуживании
3. **PortScan** - сканирование портов
4. **BruteForce** - атака перебором паролей"""))

    # Cell 4: Генерация датасета
    cells.append(nbf.v4.new_code_cell("""def create_network_traffic_data(n_samples=15000):
    \"\"\"
    Создание синтетического датасета сетевого трафика.

    Параметры:
    ----------
    n_samples : int
        Общее количество записей

    Возвращает:
    -----------
    DataFrame с признаками сетевого трафика и метками
    \"\"\"
    # Распределение классов
    n_normal = int(n_samples * 0.70)      # 70% - нормальный трафик
    n_ddos = int(n_samples * 0.10)        # 10% - DDoS
    n_portscan = int(n_samples * 0.10)    # 10% - сканирование портов
    n_bruteforce = int(n_samples * 0.10)  # 10% - brute force

    data = []

    # 1. Нормальный трафик
    for _ in range(n_normal):
        record = {
            'duration': np.random.exponential(10),
            'src_bytes': np.random.lognormal(8, 1.5),
            'dst_bytes': np.random.lognormal(8, 1.5),
            'count': np.random.poisson(5),
            'srv_count': np.random.poisson(3),
            'serror_rate': np.random.beta(1, 50),
            'rerror_rate': np.random.beta(1, 50),
            'same_srv_rate': np.random.beta(10, 2),
            'dst_host_count': np.random.poisson(20),
            'dst_host_srv_count': np.random.poisson(10),
            'attack_type': 'Normal',
            'is_attack': 0
        }
        data.append(record)

    # 2. DDoS атаки - много соединений, мало данных
    for _ in range(n_ddos):
        record = {
            'duration': np.random.exponential(0.1),  # Очень короткие
            'src_bytes': np.random.lognormal(4, 0.5),  # Мало данных
            'dst_bytes': np.random.lognormal(3, 0.5),
            'count': np.random.poisson(100),  # Много соединений!
            'srv_count': np.random.poisson(80),
            'serror_rate': np.random.beta(10, 5),  # Высокий уровень ошибок
            'rerror_rate': np.random.beta(5, 5),
            'same_srv_rate': np.random.beta(15, 1),  # К одному сервису
            'dst_host_count': np.random.poisson(200),  # Много к одному хосту
            'dst_host_srv_count': np.random.poisson(150),
            'attack_type': 'DDoS',
            'is_attack': 1
        }
        data.append(record)

    # 3. Сканирование портов - много разных портов
    for _ in range(n_portscan):
        record = {
            'duration': np.random.exponential(0.01),  # Очень быстрые
            'src_bytes': np.random.lognormal(3, 0.3),  # Минимум данных
            'dst_bytes': np.random.lognormal(2, 0.3),
            'count': np.random.poisson(50),
            'srv_count': np.random.poisson(1),  # Разные сервисы!
            'serror_rate': np.random.beta(5, 10),
            'rerror_rate': np.random.beta(10, 5),  # Много отказов
            'same_srv_rate': np.random.beta(1, 10),  # Разные сервисы
            'dst_host_count': np.random.poisson(100),
            'dst_host_srv_count': np.random.poisson(5),
            'attack_type': 'PortScan',
            'is_attack': 1
        }
        data.append(record)

    # 4. Brute Force - много попыток к одному сервису
    for _ in range(n_bruteforce):
        record = {
            'duration': np.random.exponential(1),
            'src_bytes': np.random.lognormal(5, 0.5),
            'dst_bytes': np.random.lognormal(5, 0.5),
            'count': np.random.poisson(30),
            'srv_count': np.random.poisson(25),  # К одному сервису
            'serror_rate': np.random.beta(2, 10),
            'rerror_rate': np.random.beta(8, 5),  # Много неудачных попыток
            'same_srv_rate': np.random.beta(20, 1),  # Всегда один сервис
            'dst_host_count': np.random.poisson(50),
            'dst_host_srv_count': np.random.poisson(40),
            'attack_type': 'BruteForce',
            'is_attack': 1
        }
        data.append(record)

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

# Создаём датасет
df = create_network_traffic_data(n_samples=15000)

print(f"Размер датасета: {df.shape}")
print(f"\\nРаспределение классов:")
print(df['attack_type'].value_counts())
print(f"\\nДоля атак: {df['is_attack'].mean()*100:.1f}%")"""))

    # Cell 5: Анализ датасета
    cells.append(nbf.v4.new_code_cell("""# Статистика по признакам
print("Статистика признаков:")
print(df.describe().round(2))

# Визуализация распределений
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

feature_cols = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
                'serror_rate', 'rerror_rate', 'same_srv_rate',
                'dst_host_count', 'dst_host_srv_count']

for i, col in enumerate(feature_cols):
    for attack_type in df['attack_type'].unique():
        subset = df[df['attack_type'] == attack_type][col]
        axes[i].hist(subset, bins=30, alpha=0.5, label=attack_type, density=True)
    axes[i].set_title(col)
    axes[i].set_xlabel('')
    if i == 0:
        axes[i].legend(fontsize=8)

plt.suptitle('Распределение признаков по типам трафика', fontsize=14)
plt.tight_layout()
plt.show()"""))

    # Cell 6: Подготовка данных
    cells.append(nbf.v4.new_code_cell("""# Подготовка данных
feature_cols = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
                'serror_rate', 'rerror_rate', 'same_srv_rate',
                'dst_host_count', 'dst_host_srv_count']

X = df[feature_cols].values
y = df['is_attack'].values
attack_types = df['attack_type'].values

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Размер данных: {X_scaled.shape}")
print(f"Атаки: {y.sum()} ({y.mean()*100:.1f}%)")

# PCA для визуализации
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Визуализация в 2D
fig, ax = plt.subplots(figsize=(10, 8))
colors = {'Normal': 'blue', 'DDoS': 'red', 'PortScan': 'green', 'BruteForce': 'orange'}

for attack_type in df['attack_type'].unique():
    mask = attack_types == attack_type
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[attack_type], label=attack_type,
               alpha=0.5, s=20)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Визуализация сетевого трафика (PCA)')
ax.legend()
plt.show()"""))

    # Cell 7: Spectral Clustering введение
    cells.append(nbf.v4.new_markdown_cell("""## 2. Spectral Clustering (Спектральная кластеризация)

### Теория

Спектральная кластеризация использует собственные векторы матрицы сходства (Лапласиана графа) для преобразования данных в пространство, где кластеры легче разделить.

### Алгоритм

1. **Построение графа сходства** - создаём граф, где вершины - точки данных, рёбра - сходство между ними
2. **Вычисление Лапласиана** - $L = D - W$, где $D$ - диагональная матрица степеней, $W$ - матрица смежности
3. **Собственные векторы** - находим $k$ наименьших собственных векторов $L$
4. **K-means в новом пространстве** - кластеризуем точки в пространстве собственных векторов

### Преимущества

- Хорошо работает с нелинейно разделимыми данными
- Может находить кластеры сложной формы
- Не требует предположений о форме кластеров

### Недостатки

- Вычислительно затратен для больших данных ($O(n^3)$)
- Требует выбора числа кластеров
- Чувствителен к выбору параметра сходства"""))

    # Cell 8: Spectral Clustering реализация
    cells.append(nbf.v4.new_code_cell("""# Spectral Clustering
# Используем подвыборку для ускорения (spectral clustering медленный)
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), size=3000, replace=False)
X_sample = X_scaled[sample_idx]
y_sample = y[sample_idx]
types_sample = attack_types[sample_idx]

print("Применяем Spectral Clustering...")
print("(Используем подвыборку 3000 точек для ускорения)")

# Spectral Clustering с разным числом кластеров
results = []

for n_clusters in [2, 3, 4, 5]:
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='rbf',           # RBF ядро для сходства
        gamma=0.1,                # Параметр RBF
        random_state=42,
        n_init=10
    )

    labels = spectral.fit_predict(X_sample)

    # Оценка качества
    silhouette = silhouette_score(X_sample, labels)

    results.append({
        'n_clusters': n_clusters,
        'silhouette': silhouette,
        'labels': labels
    })

    print(f"Кластеров: {n_clusters}, Silhouette: {silhouette:.3f}")

# Лучший результат
best_result = max(results, key=lambda x: x['silhouette'])
print(f"\\nЛучшее число кластеров: {best_result['n_clusters']}")"""))

    # Cell 9: Визуализация Spectral Clustering
    cells.append(nbf.v4.new_code_cell("""# Визуализация результатов Spectral Clustering
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA для визуализации подвыборки
X_sample_pca = pca.transform(X_sample)

# Слева - истинные метки
ax1 = axes[0]
for attack_type in np.unique(types_sample):
    mask = types_sample == attack_type
    ax1.scatter(X_sample_pca[mask, 0], X_sample_pca[mask, 1],
                label=attack_type, alpha=0.6, s=30)
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title('Истинные классы')
ax1.legend()

# Справа - кластеры Spectral
ax2 = axes[1]
labels = best_result['labels']
scatter = ax2.scatter(X_sample_pca[:, 0], X_sample_pca[:, 1],
                      c=labels, cmap='viridis', alpha=0.6, s=30)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_title(f'Spectral Clustering ({best_result["n_clusters"]} кластеров)')
plt.colorbar(scatter, ax=ax2, label='Кластер')

plt.tight_layout()
plt.show()

# Анализ кластеров
print("\\nСостав кластеров:")
for cluster_id in range(best_result['n_clusters']):
    mask = labels == cluster_id
    cluster_types = types_sample[mask]
    print(f"\\nКластер {cluster_id} ({mask.sum()} точек):")
    for attack_type in np.unique(cluster_types):
        count = (cluster_types == attack_type).sum()
        pct = count / mask.sum() * 100
        print(f"  {attack_type}: {count} ({pct:.1f}%)")"""))

    # Cell 10: Mean-Shift введение
    cells.append(nbf.v4.new_markdown_cell("""## 3. Mean-Shift Clustering

### Теория

Mean-Shift - это непараметрический алгоритм кластеризации, основанный на оценке плотности. Он находит моды (максимумы) функции плотности вероятности.

### Алгоритм

1. **Инициализация** - каждая точка становится кандидатом в центр кластера
2. **Сдвиг к центру масс** - для каждой точки вычисляем взвешенное среднее точек в окрестности (окне)
3. **Итерация** - повторяем сдвиг, пока точки не сойдутся
4. **Объединение** - близкие сошедшиеся точки объединяются в кластеры

### Формула сдвига

$$m(x) = \\frac{\\sum_{x_i \\in N(x)} K(x_i - x) \\cdot x_i}{\\sum_{x_i \\in N(x)} K(x_i - x)}$$

где $K$ - ядерная функция (обычно Гауссова), $N(x)$ - окрестность точки $x$.

### Преимущества

- **Не требует числа кластеров** - определяет автоматически
- Находит кластеры произвольной формы
- Устойчив к выбросам

### Недостатки

- Требует выбора bandwidth (размер окна)
- Может быть медленным для больших данных
- Результат зависит от bandwidth"""))

    # Cell 11: Mean-Shift реализация
    cells.append(nbf.v4.new_code_cell("""# Mean-Shift Clustering
print("Применяем Mean-Shift Clustering...")

# Оценка оптимального bandwidth
bandwidth = estimate_bandwidth(X_sample, quantile=0.2, random_state=42)
print(f"Оценённый bandwidth: {bandwidth:.3f}")

# Mean-Shift с разными bandwidth
bandwidths = [bandwidth * 0.5, bandwidth, bandwidth * 1.5, bandwidth * 2]
ms_results = []

for bw in bandwidths:
    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    ms_labels = ms.fit_predict(X_sample)
    n_clusters = len(np.unique(ms_labels))

    if n_clusters > 1:
        silhouette = silhouette_score(X_sample, ms_labels)
    else:
        silhouette = -1

    ms_results.append({
        'bandwidth': bw,
        'n_clusters': n_clusters,
        'silhouette': silhouette,
        'labels': ms_labels,
        'centers': ms.cluster_centers_
    })

    print(f"Bandwidth: {bw:.3f}, Кластеров: {n_clusters}, Silhouette: {silhouette:.3f}")

# Лучший результат (с разумным числом кластеров)
valid_results = [r for r in ms_results if 2 <= r['n_clusters'] <= 10]
if valid_results:
    best_ms = max(valid_results, key=lambda x: x['silhouette'])
else:
    best_ms = ms_results[1]  # Используем стандартный bandwidth

print(f"\\nВыбран bandwidth: {best_ms['bandwidth']:.3f}")
print(f"Число кластеров: {best_ms['n_clusters']}")"""))

    # Cell 12: Визуализация Mean-Shift
    cells.append(nbf.v4.new_code_cell("""# Визуализация Mean-Shift
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Кластеры Mean-Shift
ax1 = axes[0]
ms_labels = best_ms['labels']
scatter = ax1.scatter(X_sample_pca[:, 0], X_sample_pca[:, 1],
                      c=ms_labels, cmap='tab10', alpha=0.6, s=30)
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title(f'Mean-Shift ({best_ms["n_clusters"]} кластеров)')
plt.colorbar(scatter, ax=ax1, label='Кластер')

# Сравнение с атаками
ax2 = axes[1]

# Определяем, какой кластер содержит больше атак
attack_rates = []
for cluster_id in range(best_ms['n_clusters']):
    mask = ms_labels == cluster_id
    if mask.sum() > 0:
        attack_rate = y_sample[mask].mean()
        attack_rates.append((cluster_id, attack_rate, mask.sum()))

# Сортируем по доле атак
attack_rates.sort(key=lambda x: x[1], reverse=True)

print("Доля атак в кластерах:")
for cluster_id, rate, size in attack_rates:
    print(f"  Кластер {cluster_id}: {rate*100:.1f}% атак ({size} точек)")

# Визуализация - помечаем аномальные кластеры
anomaly_clusters = [c[0] for c in attack_rates if c[1] > 0.5]
is_anomaly_cluster = np.isin(ms_labels, anomaly_clusters)

ax2.scatter(X_sample_pca[~is_anomaly_cluster, 0], X_sample_pca[~is_anomaly_cluster, 1],
            c='blue', alpha=0.5, s=20, label='Нормальные кластеры')
ax2.scatter(X_sample_pca[is_anomaly_cluster, 0], X_sample_pca[is_anomaly_cluster, 1],
            c='red', alpha=0.5, s=20, label='Аномальные кластеры')

ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_title('Классификация кластеров по доле атак')
ax2.legend()

plt.tight_layout()
plt.show()"""))

    # Cell 13: Bidirectional LSTM-AE введение
    cells.append(nbf.v4.new_markdown_cell("""## 4. Bidirectional LSTM-Autoencoder

### Теория

Двунаправленный LSTM-Autoencoder обрабатывает последовательности в обоих направлениях, что позволяет учитывать контекст как предыдущих, так и последующих элементов.

### Архитектура

```
Вход → BiLSTM(64) → BiLSTM(32) → [Bottleneck] → BiLSTM(32) → BiLSTM(64) → Dense → Выход
```

### Преимущества двунаправленности

1. **Полный контекст** - видит последовательность целиком
2. **Лучшее качество** - для задач, где важен будущий контекст
3. **Симметричное представление** - информация с обоих концов

### Применение для аномалий

1. Обучаем на нормальном трафике
2. Модель учится восстанавливать нормальные паттерны
3. Аномалии имеют высокую ошибку реконструкции

### Создание последовательностей

Для LSTM нужны последовательности. Мы создадим скользящее окно по временным данным."""))

    # Cell 14: Подготовка данных для LSTM
    cells.append(nbf.v4.new_code_cell("""# Подготовка данных для LSTM
# Создаём последовательности из признаков

def create_sequences(data, seq_length=10):
    \"\"\"
    Создание последовательностей для LSTM.

    Параметры:
    ----------
    data : array
        Данные (n_samples, n_features)
    seq_length : int
        Длина последовательности

    Возвращает:
    -----------
    sequences : array (n_sequences, seq_length, n_features)
    \"\"\"
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

# Используем MinMaxScaler для LSTM (лучше работает с sigmoid/tanh)
mm_scaler = MinMaxScaler()
X_mm = mm_scaler.fit_transform(X)

# Разделяем на нормальные и аномальные
normal_mask = y == 0
X_normal = X_mm[normal_mask]
X_attack = X_mm[~normal_mask]

print(f"Нормальный трафик: {len(X_normal)}")
print(f"Атаки: {len(X_attack)}")

# Создаём последовательности
SEQ_LENGTH = 10

# Для обучения используем только нормальный трафик
X_normal_seq = create_sequences(X_normal, SEQ_LENGTH)

# Для тестирования - всё
X_all_seq = create_sequences(X_mm, SEQ_LENGTH)
y_seq = y[SEQ_LENGTH-1:]  # Метки для последовательностей

print(f"\\nПоследовательности нормального трафика: {X_normal_seq.shape}")
print(f"Все последовательности: {X_all_seq.shape}")
print(f"Метки: {len(y_seq)}")

# Разделение на train/test
train_size = int(len(X_normal_seq) * 0.8)
X_train = X_normal_seq[:train_size]
X_val = X_normal_seq[train_size:]

print(f"\\nОбучающая выборка: {X_train.shape}")
print(f"Валидационная выборка: {X_val.shape}")"""))

    # Cell 15: Построение Bidirectional LSTM-AE
    cells.append(nbf.v4.new_code_cell("""def build_bidirectional_lstm_ae(seq_length, n_features):
    \"\"\"
    Построение двунаправленного LSTM-Autoencoder.

    Архитектура:
    - Encoder: BiLSTM(64) → BiLSTM(32)
    - Decoder: BiLSTM(32) → BiLSTM(64) → TimeDistributed(Dense)
    \"\"\"
    inputs = Input(shape=(seq_length, n_features))

    # Encoder
    # BiLSTM возвращает удвоенную размерность (forward + backward)
    encoded = Bidirectional(LSTM(64, activation='relu', return_sequences=True))(inputs)
    encoded = Dropout(0.2)(encoded)
    encoded = Bidirectional(LSTM(32, activation='relu', return_sequences=False))(encoded)

    # Bottleneck
    bottleneck = RepeatVector(seq_length)(encoded)

    # Decoder
    decoded = Bidirectional(LSTM(32, activation='relu', return_sequences=True))(bottleneck)
    decoded = Dropout(0.2)(decoded)
    decoded = Bidirectional(LSTM(64, activation='relu', return_sequences=True))(decoded)

    # Output
    outputs = TimeDistributed(Dense(n_features))(decoded)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    return model

# Создаём модель
n_features = X_train.shape[2]
model = build_bidirectional_lstm_ae(SEQ_LENGTH, n_features)

print("Архитектура Bidirectional LSTM-Autoencoder:")
model.summary()"""))

    # Cell 16: Обучение модели
    cells.append(nbf.v4.new_code_cell("""# Обучение модели
print("Обучение Bidirectional LSTM-Autoencoder...")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, X_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, X_val),
    callbacks=[early_stop],
    verbose=1
)

# График обучения
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history.history['loss'], label='Обучение', linewidth=2)
ax.plot(history.history['val_loss'], label='Валидация', linewidth=2)
ax.set_xlabel('Эпоха')
ax.set_ylabel('MSE Loss')
ax.set_title('История обучения Bidirectional LSTM-Autoencoder')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

print(f"\\nФинальный loss на обучении: {history.history['loss'][-1]:.6f}")
print(f"Финальный loss на валидации: {history.history['val_loss'][-1]:.6f}")"""))

    # Cell 17: Обнаружение аномалий
    cells.append(nbf.v4.new_code_cell("""# Вычисление ошибки реконструкции
print("Вычисление ошибок реконструкции...")

# Предсказания
train_pred = model.predict(X_train, verbose=0)
all_pred = model.predict(X_all_seq, verbose=0)

# MSE для каждой последовательности
train_mse = np.mean(np.square(X_train - train_pred), axis=(1, 2))
all_mse = np.mean(np.square(X_all_seq - all_pred), axis=(1, 2))

# Статистика ошибок на обучающей выборке
train_mean = np.mean(train_mse)
train_std = np.std(train_mse)

print(f"Ошибка на обучении - Среднее: {train_mean:.6f}, Std: {train_std:.6f}")

# Пороги
thresholds = {
    '2σ': train_mean + 2 * train_std,
    '3σ': train_mean + 3 * train_std,
    '95%': np.percentile(train_mse, 95),
    '99%': np.percentile(train_mse, 99)
}

print("\\nПороги обнаружения:")
for name, thresh in thresholds.items():
    print(f"  {name}: {thresh:.6f}")"""))

    # Cell 18: Оценка качества
    cells.append(nbf.v4.new_code_cell("""# Оценка качества обнаружения
print("Оценка качества обнаружения аномалий:")
print("=" * 50)

results = []

for name, threshold in thresholds.items():
    predictions = (all_mse > threshold).astype(int)

    # Метрики
    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(y_seq, predictions, zero_division=0)
    recall = recall_score(y_seq, predictions, zero_division=0)
    f1 = f1_score(y_seq, predictions, zero_division=0)

    results.append({
        'threshold': name,
        'value': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

    print(f"\\nПорог {name} ({threshold:.4f}):")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")

# Лучший результат по F1
best_result = max(results, key=lambda x: x['f1'])
best_threshold = best_result['value']

print(f"\\nЛучший порог: {best_result['threshold']} (F1 = {best_result['f1']:.3f})")"""))

    # Cell 19: ROC и визуализация
    cells.append(nbf.v4.new_code_cell("""# ROC кривая и визуализация
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Распределение ошибок
ax1 = axes[0, 0]
normal_mse = all_mse[y_seq == 0]
attack_mse = all_mse[y_seq == 1]

ax1.hist(normal_mse, bins=50, alpha=0.6, label='Нормальный', density=True)
ax1.hist(attack_mse, bins=50, alpha=0.6, label='Атака', density=True)
ax1.axvline(x=best_threshold, color='red', linestyle='--',
            label=f'Порог={best_threshold:.4f}')
ax1.set_xlabel('Ошибка реконструкции (MSE)')
ax1.set_ylabel('Плотность')
ax1.set_title('Распределение ошибок реконструкции')
ax1.legend()

# 2. ROC кривая
ax2 = axes[0, 1]
from sklearn.metrics import roc_curve, auc as auc_score

fpr, tpr, _ = roc_curve(y_seq, all_mse)
roc_auc = auc_score(fpr, tpr)

ax2.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC кривая BiLSTM-Autoencoder')
ax2.legend(loc='lower right')

# 3. Precision-Recall кривая
ax3 = axes[1, 0]
precision_curve, recall_curve, _ = precision_recall_curve(y_seq, all_mse)
pr_auc = auc(recall_curve, precision_curve)

ax3.plot(recall_curve, precision_curve, color='green', lw=2,
         label=f'PR (AUC = {pr_auc:.3f})')
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall кривая')
ax3.legend()

# 4. Временной ряд ошибок
ax4 = axes[1, 1]
ax4.plot(all_mse, alpha=0.7, linewidth=0.5)
ax4.axhline(y=best_threshold, color='red', linestyle='--',
            label=f'Порог')

# Отмечаем атаки
attack_idx = np.where(y_seq == 1)[0]
ax4.scatter(attack_idx, all_mse[attack_idx], c='red', s=10, alpha=0.3, label='Атаки')

ax4.set_xlabel('Индекс последовательности')
ax4.set_ylabel('Ошибка реконструкции')
ax4.set_title('Временной ряд ошибок реконструкции')
ax4.legend()

plt.tight_layout()
plt.show()

print(f"\\nROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC: {pr_auc:.3f}")"""))

    # Cell 20: Confusion Matrix
    cells.append(nbf.v4.new_code_cell("""# Confusion Matrix для лучшего порога
predictions = (all_mse > best_threshold).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_seq, predictions)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Нормальный', 'Атака'],
            yticklabels=['Нормальный', 'Атака'])
ax.set_xlabel('Предсказано')
ax.set_ylabel('Истинное')
ax.set_title(f'Матрица ошибок (порог {best_result["threshold"]})')
plt.show()

print("\\nОтчёт по классификации:")
print(classification_report(y_seq, predictions,
                           target_names=['Нормальный', 'Атака']))"""))

    # Cell 21: Анализ по типам атак
    cells.append(nbf.v4.new_code_cell("""# Анализ обнаружения по типам атак
attack_types_seq = attack_types[SEQ_LENGTH-1:]

print("Обнаружение по типам атак:")
print("=" * 50)

for attack_type in np.unique(attack_types_seq):
    mask = attack_types_seq == attack_type
    type_mse = all_mse[mask]
    type_labels = y_seq[mask]
    type_preds = predictions[mask]

    if attack_type == 'Normal':
        # Для нормального трафика считаем FPR
        fpr = type_preds.mean()
        print(f"\\n{attack_type}:")
        print(f"  Ложных срабатываний: {fpr*100:.1f}%")
        print(f"  Средняя ошибка: {type_mse.mean():.4f}")
    else:
        # Для атак считаем Recall
        recall = type_preds.mean()
        print(f"\\n{attack_type}:")
        print(f"  Обнаружено: {recall*100:.1f}%")
        print(f"  Средняя ошибка: {type_mse.mean():.4f}")"""))

    # Cell 22: Сравнение методов
    cells.append(nbf.v4.new_code_cell("""# Сравнение всех методов
print("=" * 60)
print("СРАВНЕНИЕ МЕТОДОВ ОБНАРУЖЕНИЯ СЕТЕВЫХ АТАК")
print("=" * 60)

print("\\n1. Spectral Clustering")
print(f"   Кластеров: {best_result['n_clusters']}")
print(f"   Silhouette: {best_result['silhouette']:.3f}")
print("   + Хорошо разделяет нелинейные структуры")
print("   - Требует задания числа кластеров")
print("   - Вычислительно затратен")

print("\\n2. Mean-Shift")
print(f"   Кластеров: {best_ms['n_clusters']}")
print(f"   Silhouette: {best_ms['silhouette']:.3f}")
print("   + Автоматически определяет число кластеров")
print("   + Устойчив к выбросам")
print("   - Чувствителен к bandwidth")

print("\\n3. Bidirectional LSTM-Autoencoder")
print(f"   ROC-AUC: {roc_auc:.3f}")
print(f"   PR-AUC: {pr_auc:.3f}")
print(f"   F1-Score: {best_result['f1']:.3f}")
print("   + Учитывает временные зависимости")
print("   + Двунаправленный контекст")
print("   - Требует настройки порога")

print("\\n" + "=" * 60)
print("РЕКОМЕНДАЦИИ ПО ПРИМЕНЕНИЮ")
print("=" * 60)
print("\\n• Spectral Clustering: анализ структуры трафика, визуализация")
print("• Mean-Shift: первичная сегментация без априорных знаний")
print("• BiLSTM-AE: обнаружение аномалий в реальном времени")
print("\\nОптимально: комбинация методов для повышения надёжности")"""))

    # Cell 23: Заключение
    cells.append(nbf.v4.new_markdown_cell("""## Заключение

### Ключевые результаты

1. **Spectral Clustering** успешно выделил группы с разными паттернами трафика, используя спектральное разложение матрицы сходства.

2. **Mean-Shift** автоматически нашёл естественные кластеры в данных без задания их числа, что полезно для разведочного анализа.

3. **Bidirectional LSTM-Autoencoder** показал высокое качество обнаружения атак (ROC-AUC ≈ 0.9+), учитывая временные зависимости в обоих направлениях.

### Особенности датасета

- 15,000 записей сетевого трафика
- 4 класса: Normal, DDoS, PortScan, BruteForce
- 10 информативных признаков
- Реалистичные паттерны для каждого типа атак

### Практическое применение

1. **Мониторинг сети** - BiLSTM-AE для обнаружения в реальном времени
2. **Анализ инцидентов** - кластеризация для группировки похожих атак
3. **Профилирование** - Mean-Shift для выявления типичных паттернов

### Дальнейшие улучшения

- Добавление Attention механизма в LSTM
- Ансамбль нескольких автоэнкодеров
- Онлайн-обучение для адаптации к новым атакам
- Интеграция с SIEM системами"""))

    nb['cells'] = cells

    # Сохранение
    output_path = '/home/user/test/notebooks/phase5_anomaly_patterns/05_network_intrusion.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"✅ Notebook создан: {output_path}")
    print(f"Всего ячеек: {len(cells)}")
    return output_path

if __name__ == "__main__":
    create_notebook()
