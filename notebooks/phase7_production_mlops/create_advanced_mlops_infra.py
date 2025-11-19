#!/usr/bin/env python3
"""
Скрипт для создания notebook по продвинутой MLOps инфраструктуре.
Методы: Kubernetes deployment, Feature Store, Data Drift Detection
"""

import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Заголовок
    cells.append(nbf.v4.new_markdown_cell("""# Фаза 7.3: Продвинутая MLOps инфраструктура

## Kubernetes, Feature Store и мониторинг дрифта

В этом ноутбуке мы рассмотрим продвинутые практики MLOps:

### Темы

1. **Kubernetes Deployment** - манифесты для развёртывания ML-моделей в K8s
2. **Feature Store** - паттерн для управления признаками
3. **Data Drift Detection** - обнаружение изменений в данных (стиль Evidently)

### Задача

Продолжаем работу с моделью оттока клиентов (Customer Churn) и показываем, как:
- Развернуть модель в Kubernetes
- Организовать хранение и переиспользование признаков
- Мониторить данные на предмет дрифта

### Практическая ценность

Эти практики необходимы для production-ready ML систем, обеспечивая масштабируемость, надёжность и качество."""))

    # Cell 2: Импорты
    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from scipy import stats
from datetime import datetime, timedelta
import json
import yaml
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("Библиотеки загружены успешно")"""))

    # Cell 3: Создание данных
    cells.append(nbf.v4.new_code_cell("""# Создаём датасет оттока клиентов (расширенный)
def create_churn_data(n_samples=10000, drift=False, drift_magnitude=0.3):
    \"\"\"
    Создание датасета оттока с возможностью добавления дрифта.

    Параметры:
    ----------
    n_samples : int
        Количество клиентов
    drift : bool
        Добавить ли дрифт в данные
    drift_magnitude : float
        Сила дрифта

    Возвращает:
    -----------
    DataFrame с данными клиентов
    \"\"\"
    # Базовые признаки
    data = {
        'customer_id': range(1, n_samples + 1),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 100, n_samples),
        'total_charges': np.zeros(n_samples),
        'num_support_tickets': np.random.poisson(2, n_samples),
        'num_referrals': np.random.poisson(1, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                          n_samples, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Electronic', 'Mailed', 'Bank', 'Credit'],
                                           n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber', 'No'],
                                             n_samples, p=[0.35, 0.45, 0.2]),
    }

    # Добавляем дрифт если нужно
    if drift:
        # Дрифт: увеличение цен и количества тикетов
        data['monthly_charges'] *= (1 + drift_magnitude)
        data['num_support_tickets'] = np.random.poisson(3.5, n_samples)
        # Изменение распределения контрактов
        data['contract_type'] = np.random.choice(
            ['Month-to-month', 'One year', 'Two year'],
            n_samples, p=[0.65, 0.25, 0.10]
        )

    # Вычисляем total_charges
    data['total_charges'] = data['tenure_months'] * data['monthly_charges'] * \\
                            np.random.uniform(0.9, 1.1, n_samples)

    # Генерируем отток
    churn_prob = 0.1 + \\
        0.3 * (np.array([1 if c == 'Month-to-month' else 0 for c in data['contract_type']])) + \\
        0.1 * (data['tenure_months'] < 12) / 12 + \\
        0.1 * (data['num_support_tickets'] > 3) + \\
        0.1 * (data['monthly_charges'] > 70)

    churn_prob = np.clip(churn_prob, 0, 1)
    data['churned'] = np.random.binomial(1, churn_prob)

    df = pd.DataFrame(data)
    return df

# Создаём обучающие данные и данные с дрифтом
df_train = create_churn_data(n_samples=10000, drift=False)
df_production = create_churn_data(n_samples=3000, drift=True, drift_magnitude=0.3)

print(f"Обучающие данные: {df_train.shape}")
print(f"Production данные (с дрифтом): {df_production.shape}")
print(f"\\nОтток в обучении: {df_train['churned'].mean()*100:.1f}%")
print(f"Отток в production: {df_production['churned'].mean()*100:.1f}%")"""))

    # Cell 4: Kubernetes введение
    cells.append(nbf.v4.new_markdown_cell("""## 1. Kubernetes Deployment

### Что такое Kubernetes?

Kubernetes (K8s) - это платформа для оркестрации контейнеров, которая:
- Автоматически масштабирует приложения
- Обеспечивает высокую доступность
- Управляет развёртыванием и откатом

### Компоненты для ML

1. **Deployment** - описывает желаемое состояние приложения
2. **Service** - обеспечивает сетевой доступ
3. **ConfigMap** - хранит конфигурацию
4. **Secret** - хранит sensitive данные
5. **HorizontalPodAutoscaler** - автоматическое масштабирование

### Преимущества для ML

- **Масштабируемость** - легко добавить реплики при нагрузке
- **Воспроизводимость** - окружение описано в манифестах
- **Изоляция** - каждая модель в своём контейнере
- **Rolling updates** - обновление без простоя"""))

    # Cell 5: Kubernetes манифесты
    cells.append(nbf.v4.new_code_cell("""# Генерация Kubernetes манифестов для ML модели

def generate_k8s_manifests(model_name, image, replicas=3, cpu_limit='500m', memory_limit='512Mi'):
    \"\"\"
    Генерация Kubernetes манифестов для развёртывания ML модели.

    Параметры:
    ----------
    model_name : str
        Название модели
    image : str
        Docker образ
    replicas : int
        Число реплик
    cpu_limit : str
        Лимит CPU
    memory_limit : str
        Лимит памяти

    Возвращает:
    -----------
    dict с манифестами
    \"\"\"

    # 1. Deployment
    deployment = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': f'{model_name}-deployment',
            'labels': {
                'app': model_name,
                'version': 'v1'
            }
        },
        'spec': {
            'replicas': replicas,
            'selector': {
                'matchLabels': {
                    'app': model_name
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': model_name,
                        'version': 'v1'
                    },
                    'annotations': {
                        'prometheus.io/scrape': 'true',
                        'prometheus.io/port': '8000'
                    }
                },
                'spec': {
                    'containers': [{
                        'name': model_name,
                        'image': image,
                        'ports': [{'containerPort': 8000}],
                        'resources': {
                            'requests': {
                                'cpu': '100m',
                                'memory': '256Mi'
                            },
                            'limits': {
                                'cpu': cpu_limit,
                                'memory': memory_limit
                            }
                        },
                        'env': [
                            {'name': 'MODEL_NAME', 'value': model_name},
                            {'name': 'LOG_LEVEL', 'value': 'INFO'}
                        ],
                        'livenessProbe': {
                            'httpGet': {
                                'path': '/health',
                                'port': 8000
                            },
                            'initialDelaySeconds': 30,
                            'periodSeconds': 10
                        },
                        'readinessProbe': {
                            'httpGet': {
                                'path': '/ready',
                                'port': 8000
                            },
                            'initialDelaySeconds': 5,
                            'periodSeconds': 5
                        }
                    }]
                }
            }
        }
    }

    # 2. Service
    service = {
        'apiVersion': 'v1',
        'kind': 'Service',
        'metadata': {
            'name': f'{model_name}-service'
        },
        'spec': {
            'selector': {
                'app': model_name
            },
            'ports': [{
                'protocol': 'TCP',
                'port': 80,
                'targetPort': 8000
            }],
            'type': 'ClusterIP'
        }
    }

    # 3. HorizontalPodAutoscaler
    hpa = {
        'apiVersion': 'autoscaling/v2',
        'kind': 'HorizontalPodAutoscaler',
        'metadata': {
            'name': f'{model_name}-hpa'
        },
        'spec': {
            'scaleTargetRef': {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'name': f'{model_name}-deployment'
            },
            'minReplicas': 2,
            'maxReplicas': 10,
            'metrics': [{
                'type': 'Resource',
                'resource': {
                    'name': 'cpu',
                    'target': {
                        'type': 'Utilization',
                        'averageUtilization': 70
                    }
                }
            }]
        }
    }

    # 4. ConfigMap для конфигурации модели
    configmap = {
        'apiVersion': 'v1',
        'kind': 'ConfigMap',
        'metadata': {
            'name': f'{model_name}-config'
        },
        'data': {
            'model_config.yaml': yaml.dump({
                'model_name': model_name,
                'version': 'v1',
                'threshold': 0.5,
                'features': ['tenure_months', 'monthly_charges', 'total_charges',
                            'num_support_tickets', 'contract_type']
            })
        }
    }

    return {
        'deployment': deployment,
        'service': service,
        'hpa': hpa,
        'configmap': configmap
    }

# Генерируем манифесты
manifests = generate_k8s_manifests(
    model_name='churn-predictor',
    image='ml-registry/churn-model:v1.0',
    replicas=3
)

print("Kubernetes Deployment манифест:")
print("=" * 50)
print(yaml.dump(manifests['deployment'], default_flow_style=False))"""))

    # Cell 6: Остальные манифесты
    cells.append(nbf.v4.new_code_cell("""# Service манифест
print("Kubernetes Service манифест:")
print("=" * 50)
print(yaml.dump(manifests['service'], default_flow_style=False))

print("\\nHorizontalPodAutoscaler манифест:")
print("=" * 50)
print(yaml.dump(manifests['hpa'], default_flow_style=False))"""))

    # Cell 7: Сохранение манифестов
    cells.append(nbf.v4.new_code_cell("""# Сохранение манифестов в файлы
import os

k8s_dir = '/home/user/test/notebooks/phase7_production_mlops/k8s'
os.makedirs(k8s_dir, exist_ok=True)

for name, manifest in manifests.items():
    filepath = os.path.join(k8s_dir, f'{name}.yaml')
    with open(filepath, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False)
    print(f"Сохранено: {filepath}")

print("\\nКоманды для развёртывания:")
print("kubectl apply -f k8s/")
print("kubectl get pods -l app=churn-predictor")
print("kubectl get hpa")"""))

    # Cell 8: Feature Store введение
    cells.append(nbf.v4.new_markdown_cell("""## 2. Feature Store

### Что такое Feature Store?

Feature Store - это централизованное хранилище признаков для ML, которое:
- Обеспечивает переиспользование признаков между моделями
- Гарантирует консистентность между обучением и inference
- Версионирует признаки
- Обеспечивает point-in-time корректность

### Компоненты

1. **Feature Registry** - метаданные о признаках
2. **Offline Store** - исторические данные для обучения
3. **Online Store** - низколатентное хранилище для inference
4. **Feature Transformation** - вычисление признаков

### Преимущества

- **Переиспользование** - одни признаки для многих моделей
- **Консистентность** - одинаковые признаки в train и serve
- **Версионирование** - отслеживание изменений
- **Мониторинг** - статистика по признакам

Мы реализуем упрощённую версию Feature Store."""))

    # Cell 9: Реализация Feature Store
    cells.append(nbf.v4.new_code_cell("""class SimpleFeatureStore:
    \"\"\"
    Упрощённая реализация Feature Store.

    Демонстрирует ключевые концепции:
    - Регистрация признаков
    - Хранение исторических значений
    - Получение признаков для inference
    - Статистика по признакам
    \"\"\"

    def __init__(self):
        self.feature_registry = {}  # Метаданные о признаках
        self.offline_store = {}     # Исторические данные
        self.online_store = {}      # Последние значения
        self.statistics = {}        # Статистика

    def register_feature(self, name, dtype, description, transformation=None):
        \"\"\"
        Регистрация нового признака.

        Параметры:
        ----------
        name : str
            Название признака
        dtype : str
            Тип данных
        description : str
            Описание
        transformation : callable
            Функция преобразования
        \"\"\"
        self.feature_registry[name] = {
            'name': name,
            'dtype': dtype,
            'description': description,
            'transformation': transformation,
            'created_at': datetime.now().isoformat(),
            'version': 1
        }

        self.offline_store[name] = []
        self.online_store[name] = {}
        self.statistics[name] = {
            'count': 0,
            'mean': 0,
            'std': 0,
            'min': float('inf'),
            'max': float('-inf')
        }

        print(f"Зарегистрирован признак: {name}")

    def ingest(self, entity_id, features, timestamp=None):
        \"\"\"
        Добавление значений признаков.

        Параметры:
        ----------
        entity_id : str/int
            ID сущности (например, customer_id)
        features : dict
            Словарь {название: значение}
        timestamp : datetime
            Время записи
        \"\"\"
        if timestamp is None:
            timestamp = datetime.now()

        for name, value in features.items():
            if name not in self.feature_registry:
                raise ValueError(f"Признак {name} не зарегистрирован")

            # Применяем трансформацию если есть
            transform = self.feature_registry[name].get('transformation')
            if transform:
                value = transform(value)

            # Offline store (историческое)
            self.offline_store[name].append({
                'entity_id': entity_id,
                'value': value,
                'timestamp': timestamp
            })

            # Online store (последнее значение)
            self.online_store[name][entity_id] = value

            # Обновляем статистику
            self._update_statistics(name, value)

    def _update_statistics(self, name, value):
        \"\"\"Обновление статистики признака.\"\"\"
        stats = self.statistics[name]

        # Инкрементальное обновление
        n = stats['count']
        old_mean = stats['mean']

        stats['count'] = n + 1
        stats['mean'] = old_mean + (value - old_mean) / (n + 1)

        if n > 0:
            stats['std'] = np.sqrt(
                ((n - 1) * stats['std']**2 + (value - old_mean) * (value - stats['mean'])) / n
            )

        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)

    def get_online_features(self, entity_id, feature_names):
        \"\"\"
        Получение признаков для inference (низкая латентность).

        Параметры:
        ----------
        entity_id : str/int
            ID сущности
        feature_names : list
            Список названий признаков

        Возвращает:
        -----------
        dict с признаками
        \"\"\"
        result = {}
        for name in feature_names:
            if name in self.online_store:
                result[name] = self.online_store[name].get(entity_id)
            else:
                result[name] = None
        return result

    def get_historical_features(self, entity_ids, feature_names, start_time=None, end_time=None):
        \"\"\"
        Получение исторических признаков для обучения.

        Параметры:
        ----------
        entity_ids : list
            Список ID сущностей
        feature_names : list
            Список названий признаков
        start_time : datetime
            Начало периода
        end_time : datetime
            Конец периода

        Возвращает:
        -----------
        DataFrame с историческими признаками
        \"\"\"
        records = []

        for entity_id in entity_ids:
            record = {'entity_id': entity_id}

            for name in feature_names:
                # Находим последнее значение для entity_id
                values = [
                    r for r in self.offline_store.get(name, [])
                    if r['entity_id'] == entity_id
                ]

                if start_time:
                    values = [v for v in values if v['timestamp'] >= start_time]
                if end_time:
                    values = [v for v in values if v['timestamp'] <= end_time]

                if values:
                    # Берём последнее значение
                    record[name] = sorted(values, key=lambda x: x['timestamp'])[-1]['value']
                else:
                    record[name] = None

            records.append(record)

        return pd.DataFrame(records)

    def get_feature_statistics(self, name):
        \"\"\"Получение статистики по признаку.\"\"\"
        return self.statistics.get(name, {})

    def list_features(self):
        \"\"\"Список всех зарегистрированных признаков.\"\"\"
        return list(self.feature_registry.keys())

print("Класс SimpleFeatureStore определён")"""))

    # Cell 10: Использование Feature Store
    cells.append(nbf.v4.new_code_cell("""# Создаём и настраиваем Feature Store
fs = SimpleFeatureStore()

# Регистрируем признаки
fs.register_feature(
    name='tenure_months',
    dtype='int',
    description='Количество месяцев с компанией'
)

fs.register_feature(
    name='monthly_charges',
    dtype='float',
    description='Ежемесячный платёж'
)

fs.register_feature(
    name='total_charges',
    dtype='float',
    description='Общая сумма платежей'
)

fs.register_feature(
    name='support_tickets_normalized',
    dtype='float',
    description='Нормализованное число тикетов поддержки',
    transformation=lambda x: np.log1p(x)  # log-трансформация
)

fs.register_feature(
    name='avg_monthly_charge',
    dtype='float',
    description='Средний месячный платёж (total/tenure)'
)

print("\\nЗарегистрированные признаки:")
for name in fs.list_features():
    info = fs.feature_registry[name]
    print(f"  {name}: {info['description']}")"""))

    # Cell 11: Загрузка данных в Feature Store
    cells.append(nbf.v4.new_code_cell("""# Загружаем данные в Feature Store
print("Загрузка данных в Feature Store...")

# Берём подмножество для демонстрации
sample_data = df_train.head(1000)

for _, row in sample_data.iterrows():
    customer_id = row['customer_id']

    # Вычисляем признаки
    features = {
        'tenure_months': row['tenure_months'],
        'monthly_charges': row['monthly_charges'],
        'total_charges': row['total_charges'],
        'support_tickets_normalized': row['num_support_tickets'],
        'avg_monthly_charge': row['total_charges'] / max(row['tenure_months'], 1)
    }

    fs.ingest(customer_id, features)

print(f"Загружено {len(sample_data)} записей")

# Статистика по признакам
print("\\nСтатистика признаков:")
for name in fs.list_features():
    stats = fs.get_feature_statistics(name)
    print(f"\\n{name}:")
    print(f"  Count: {stats['count']}")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Min: {stats['min']:.2f}")
    print(f"  Max: {stats['max']:.2f}")"""))

    # Cell 12: Получение признаков
    cells.append(nbf.v4.new_code_cell("""# Демонстрация получения признаков

# 1. Online features (для inference)
customer_id = 42
online_features = fs.get_online_features(
    customer_id,
    ['tenure_months', 'monthly_charges', 'support_tickets_normalized']
)

print("Online Features для клиента", customer_id)
print(json.dumps(online_features, indent=2, default=str))

# 2. Historical features (для обучения)
entity_ids = [1, 2, 3, 4, 5]
historical_df = fs.get_historical_features(
    entity_ids,
    ['tenure_months', 'monthly_charges', 'total_charges', 'avg_monthly_charge']
)

print("\\nHistorical Features:")
print(historical_df)"""))

    # Cell 13: Drift Detection введение
    cells.append(nbf.v4.new_markdown_cell("""## 3. Data Drift Detection

### Что такое Data Drift?

Data Drift - это изменение статистических свойств данных во времени. Это важно для ML, потому что:
- Модель обучена на определённом распределении
- Если production данные отличаются, качество падает
- Нужен мониторинг для раннего обнаружения

### Типы дрифта

1. **Feature Drift** - изменение распределения признаков
2. **Label Drift** - изменение распределения целевой переменной
3. **Concept Drift** - изменение связи между признаками и целью

### Методы обнаружения

1. **Статистические тесты**
   - Kolmogorov-Smirnov (K-S test)
   - Chi-squared test
   - Population Stability Index (PSI)

2. **Расстояния между распределениями**
   - Wasserstein distance
   - Jensen-Shannon divergence
   - KL divergence

Мы реализуем детектор дрифта в стиле Evidently."""))

    # Cell 14: Реализация Drift Detector
    cells.append(nbf.v4.new_code_cell("""class DriftDetector:
    \"\"\"
    Детектор дрифта данных в стиле Evidently.

    Обнаруживает изменения в распределении признаков
    между референсным и текущим датасетами.
    \"\"\"

    def __init__(self, reference_data, feature_names, categorical_features=None):
        \"\"\"
        Инициализация детектора.

        Параметры:
        ----------
        reference_data : DataFrame
            Референсные данные (обычно обучающие)
        feature_names : list
            Список числовых признаков
        categorical_features : list
            Список категориальных признаков
        \"\"\"
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []

        # Вычисляем референсную статистику
        self.reference_stats = self._compute_statistics(reference_data)

    def _compute_statistics(self, data):
        \"\"\"Вычисление статистики для датасета.\"\"\"
        stats = {}
        for col in self.feature_names:
            if col in data.columns:
                values = data[col].dropna()
                stats[col] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'median': values.median(),
                    'min': values.min(),
                    'max': values.max(),
                    'percentiles': np.percentile(values, [25, 50, 75]).tolist()
                }
        return stats

    def ks_test(self, reference, current):
        \"\"\"
        Kolmogorov-Smirnov тест для сравнения распределений.

        Возвращает:
        -----------
        statistic : float
            K-S статистика
        p_value : float
            p-value
        \"\"\"
        statistic, p_value = stats.ks_2samp(reference, current)
        return statistic, p_value

    def psi(self, reference, current, bins=10):
        \"\"\"
        Population Stability Index.

        PSI < 0.1 - нет изменений
        PSI 0.1-0.2 - небольшие изменения
        PSI > 0.2 - значительные изменения
        \"\"\"
        # Создаём bins на основе reference
        _, bin_edges = np.histogram(reference, bins=bins)

        # Считаем доли в каждом bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Нормализуем
        ref_pct = ref_counts / len(reference)
        cur_pct = cur_counts / len(current)

        # Добавляем малое число для избежания log(0)
        ref_pct = np.clip(ref_pct, 0.0001, 1)
        cur_pct = np.clip(cur_pct, 0.0001, 1)

        # PSI
        psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return psi_value

    def detect_drift(self, current_data, threshold_pvalue=0.05, threshold_psi=0.1):
        \"\"\"
        Обнаружение дрифта в текущих данных.

        Параметры:
        ----------
        current_data : DataFrame
            Текущие данные для проверки
        threshold_pvalue : float
            Порог p-value для K-S теста
        threshold_psi : float
            Порог PSI

        Возвращает:
        -----------
        report : dict
            Отчёт о дрифте
        \"\"\"
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_reference': len(self.reference_data),
            'n_current': len(current_data),
            'features': {},
            'overall_drift': False,
            'drifted_features': []
        }

        for col in self.feature_names:
            if col not in current_data.columns:
                continue

            ref_values = self.reference_data[col].dropna().values
            cur_values = current_data[col].dropna().values

            # K-S тест
            ks_stat, ks_pvalue = self.ks_test(ref_values, cur_values)

            # PSI
            psi_value = self.psi(ref_values, cur_values)

            # Изменение среднего
            ref_mean = ref_values.mean()
            cur_mean = cur_values.mean()
            mean_change = (cur_mean - ref_mean) / (ref_mean + 1e-10) * 100

            # Определяем наличие дрифта
            is_drifted = (ks_pvalue < threshold_pvalue) or (psi_value > threshold_psi)

            feature_report = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'psi': psi_value,
                'reference_mean': ref_mean,
                'current_mean': cur_mean,
                'mean_change_pct': mean_change,
                'is_drifted': is_drifted
            }

            report['features'][col] = feature_report

            if is_drifted:
                report['drifted_features'].append(col)

        report['overall_drift'] = len(report['drifted_features']) > 0

        return report

    def plot_drift_report(self, report, figsize=(14, 10)):
        \"\"\"Визуализация отчёта о дрифте.\"\"\"
        features = list(report['features'].keys())
        n_features = len(features)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. PSI по признакам
        ax1 = axes[0, 0]
        psi_values = [report['features'][f]['psi'] for f in features]
        colors = ['red' if p > 0.1 else 'orange' if p > 0.05 else 'green' for p in psi_values]
        bars = ax1.barh(features, psi_values, color=colors)
        ax1.axvline(x=0.1, color='red', linestyle='--', label='Критический (0.1)')
        ax1.axvline(x=0.05, color='orange', linestyle='--', label='Предупреждение (0.05)')
        ax1.set_xlabel('PSI')
        ax1.set_title('Population Stability Index')
        ax1.legend()

        # 2. K-S p-values
        ax2 = axes[0, 1]
        pvalues = [report['features'][f]['ks_pvalue'] for f in features]
        colors = ['red' if p < 0.05 else 'green' for p in pvalues]
        ax2.barh(features, pvalues, color=colors)
        ax2.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
        ax2.set_xlabel('p-value')
        ax2.set_title('Kolmogorov-Smirnov Test p-values')
        ax2.legend()

        # 3. Изменение среднего
        ax3 = axes[1, 0]
        mean_changes = [report['features'][f]['mean_change_pct'] for f in features]
        colors = ['red' if abs(m) > 20 else 'orange' if abs(m) > 10 else 'green' for m in mean_changes]
        ax3.barh(features, mean_changes, color=colors)
        ax3.axvline(x=0, color='black', linewidth=0.5)
        ax3.set_xlabel('Изменение среднего (%)')
        ax3.set_title('Относительное изменение среднего')

        # 4. Сводка
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_text = f\"\"\"
        СВОДКА ПО ДРИФТУ

        Референсные данные: {report['n_reference']} записей
        Текущие данные: {report['n_current']} записей

        Общий дрифт: {'ДА' if report['overall_drift'] else 'НЕТ'}

        Признаки с дрифтом:
        {chr(10).join(['  • ' + f for f in report['drifted_features']]) if report['drifted_features'] else '  Нет'}

        Рекомендации:
        {'• Требуется переобучение модели' if report['overall_drift'] else '• Модель актуальна'}
        {'• Проверить источники данных' if report['overall_drift'] else ''}
        \"\"\"

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')

        plt.suptitle('Отчёт о дрифте данных', fontsize=14)
        plt.tight_layout()
        plt.show()

print("Класс DriftDetector определён")"""))

    # Cell 15: Применение Drift Detection
    cells.append(nbf.v4.new_code_cell("""# Применяем Drift Detection

# Числовые признаки для анализа
numeric_features = ['tenure_months', 'monthly_charges', 'total_charges',
                   'num_support_tickets', 'num_referrals']

# Создаём детектор
detector = DriftDetector(
    reference_data=df_train,
    feature_names=numeric_features
)

# Обнаруживаем дрифт в production данных
print("Анализ дрифта...")
drift_report = detector.detect_drift(df_production)

# Выводим результаты
print("\\nРезультаты обнаружения дрифта:")
print("=" * 50)

for feature, metrics in drift_report['features'].items():
    status = "🔴 ДРИФТ" if metrics['is_drifted'] else "🟢 OK"
    print(f"\\n{feature} {status}")
    print(f"  PSI: {metrics['psi']:.4f}")
    print(f"  K-S p-value: {metrics['ks_pvalue']:.4f}")
    print(f"  Изменение среднего: {metrics['mean_change_pct']:+.1f}%")

print(f"\\nОбщий результат: {'ДРИФТ ОБНАРУЖЕН' if drift_report['overall_drift'] else 'Дрифт не обнаружен'}")
print(f"Признаки с дрифтом: {drift_report['drifted_features']}")"""))

    # Cell 16: Визуализация дрифта
    cells.append(nbf.v4.new_code_cell("""# Визуализация отчёта
detector.plot_drift_report(drift_report)"""))

    # Cell 17: Детальный анализ
    cells.append(nbf.v4.new_code_cell("""# Детальное сравнение распределений
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    ax = axes[i]

    # Референсные данные
    ax.hist(df_train[col], bins=30, alpha=0.5, label='Reference', density=True)
    # Production данные
    ax.hist(df_production[col], bins=30, alpha=0.5, label='Production', density=True)

    # Статистика
    ref_mean = df_train[col].mean()
    prod_mean = df_production[col].mean()

    ax.axvline(ref_mean, color='blue', linestyle='--', linewidth=2)
    ax.axvline(prod_mean, color='orange', linestyle='--', linewidth=2)

    psi = drift_report['features'][col]['psi']
    ax.set_title(f'{col}\\nPSI={psi:.3f}')
    ax.legend()

# Удаляем лишний subplot
axes[-1].axis('off')

plt.suptitle('Сравнение распределений: Reference vs Production', fontsize=14)
plt.tight_layout()
plt.show()"""))

    # Cell 18: Автоматический мониторинг
    cells.append(nbf.v4.new_code_cell("""# Пример автоматического мониторинга дрифта

def monitor_data_quality(detector, new_data, alert_threshold=0.15):
    \"\"\"
    Автоматический мониторинг качества данных.

    Параметры:
    ----------
    detector : DriftDetector
        Детектор дрифта
    new_data : DataFrame
        Новые данные
    alert_threshold : float
        Порог PSI для алерта

    Возвращает:
    -----------
    alerts : list
        Список алертов
    \"\"\"
    report = detector.detect_drift(new_data)

    alerts = []

    for feature, metrics in report['features'].items():
        if metrics['psi'] > alert_threshold:
            alert = {
                'type': 'CRITICAL',
                'feature': feature,
                'message': f'Высокий PSI ({metrics["psi"]:.3f})',
                'action': 'Требуется переобучение'
            }
            alerts.append(alert)
        elif metrics['psi'] > alert_threshold / 2:
            alert = {
                'type': 'WARNING',
                'feature': feature,
                'message': f'Повышенный PSI ({metrics["psi"]:.3f})',
                'action': 'Мониторить ситуацию'
            }
            alerts.append(alert)

    return alerts

# Запускаем мониторинг
alerts = monitor_data_quality(detector, df_production)

print("Алерты мониторинга:")
print("=" * 50)

for alert in alerts:
    icon = "🔴" if alert['type'] == 'CRITICAL' else "🟡"
    print(f"\\n{icon} {alert['type']}: {alert['feature']}")
    print(f"   {alert['message']}")
    print(f"   Действие: {alert['action']}")

if not alerts:
    print("\\n🟢 Все показатели в норме")"""))

    # Cell 19: Сравнение методов
    cells.append(nbf.v4.new_code_cell("""# Итоговое сравнение
print("=" * 60)
print("ПРОДВИНУТАЯ MLOPS ИНФРАСТРУКТУРА - ИТОГИ")
print("=" * 60)

print("\\n1. Kubernetes Deployment")
print("   Компоненты: Deployment, Service, HPA, ConfigMap")
print("   Возможности:")
print("   • Автомасштабирование по CPU")
print("   • Health checks (liveness/readiness)")
print("   • Rolling updates без простоя")
print("   • Prometheus метрики")

print("\\n2. Feature Store")
print("   Компоненты: Registry, Offline/Online Store")
print("   Возможности:")
print("   • Регистрация и версионирование признаков")
print("   • Трансформации при записи")
print("   • Быстрый доступ для inference")
print("   • Статистика по признакам")

print("\\n3. Drift Detection")
print("   Методы: K-S тест, PSI")
print("   Возможности:")
print("   • Обнаружение изменений распределения")
print("   • Автоматические алерты")
print("   • Визуализация дрифта")
print("   • Рекомендации по действиям")

print("\\n" + "=" * 60)
print("РЕКОМЕНДАЦИИ ДЛЯ PRODUCTION")
print("=" * 60)
print("\\n• Используйте Kubernetes для масштабируемости")
print("• Внедрите Feature Store для консистентности")
print("• Настройте мониторинг дрифта с алертами")
print("• Автоматизируйте переобучение при дрифте")
print("• Ведите логи всех предсказаний")"""))

    # Cell 20: Заключение
    cells.append(nbf.v4.new_markdown_cell("""## Заключение

### Ключевые результаты

1. **Kubernetes Deployment** - создали полный набор манифестов для развёртывания ML модели с автомасштабированием, health checks и метриками Prometheus.

2. **Feature Store** - реализовали упрощённую версию, демонстрирующую ключевые концепции: регистрацию признаков, хранение и получение данных для обучения и inference.

3. **Drift Detection** - обнаружили значительный дрифт в production данных по признакам monthly_charges и num_support_tickets.

### Обнаруженный дрифт

Production данные показали:
- Увеличение monthly_charges на ~30%
- Рост числа тикетов поддержки
- Смещение к месячным контрактам

Это требует переобучения модели!

### Практическое применение

1. **CI/CD для ML** - автоматическое тестирование и развёртывание моделей
2. **Мониторинг** - раннее обнаружение деградации качества
3. **Автоматизация** - триггеры на переобучение при дрифте

### Инструменты для изучения

- **Kubernetes**: minikube, kind для локальной разработки
- **Feature Store**: Feast, Tecton, Hopsworks
- **Drift Detection**: Evidently, WhyLabs, Arize

### Следующие шаги

- Интеграция с CI/CD (GitHub Actions, GitLab CI)
- Добавление A/B тестирования в production
- Настройка автоматического переобучения
- Создание дашбордов мониторинга"""))

    nb['cells'] = cells

    # Сохранение
    output_path = '/home/user/test/notebooks/phase7_production_mlops/03_advanced_mlops_infra.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"✅ Notebook создан: {output_path}")
    print(f"Всего ячеек: {len(cells)}")
    return output_path

if __name__ == "__main__":
    create_notebook()
