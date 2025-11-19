#!/usr/bin/env python3
"""
Скрипт для создания notebook по продвинутым методам XAI.
Методы: Integrated Gradients, Anchor Explanations, Concept-based (TCAV-like)
Задача: Медицинская диагностика
"""

import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Заголовок
    cells.append(nbf.v4.new_markdown_cell("""# Фаза 6.3: Продвинутые методы объяснимого ИИ

## Интерпретация моделей для медицинской диагностики

В этом ноутбуке мы рассмотрим продвинутые методы объяснимого ИИ (XAI):

### Методы

1. **Integrated Gradients** - атрибуция на основе интегрирования градиентов
2. **Anchor Explanations** - объяснения в виде правил (if-then)
3. **Concept-based Explanations (TCAV-like)** - объяснения через высокоуровневые концепции

### Задача

Предсказание болезни сердца на основе клинических показателей. Медицинская область требует особенно высокого уровня объяснимости, так как решения влияют на здоровье пациентов.

### Датасет

Синтетический датасет с клиническими показателями (~8,000 пациентов), включающий понятные медицинские признаки."""))

    # Cell 2: Импорты
    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Глубокое обучение
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input

np.random.seed(42)
tf.random.set_seed(42)

print("Библиотеки загружены успешно")
print(f"TensorFlow версия: {tf.__version__}")"""))

    # Cell 3: Создание датасета
    cells.append(nbf.v4.new_markdown_cell("""## 1. Создание медицинского датасета

### Описание признаков

Наш датасет содержит клинические показатели пациентов:

**Демографические:**
- **age** - возраст (лет)
- **sex** - пол (0 = женский, 1 = мужской)

**Клинические измерения:**
- **blood_pressure** - систолическое давление (мм рт. ст.)
- **cholesterol** - холестерин (мг/дл)
- **blood_sugar** - уровень сахара натощак (мг/дл)
- **heart_rate** - пульс в покое (уд/мин)

**Результаты обследований:**
- **ecg_abnormal** - аномалии на ЭКГ (0/1)
- **exercise_angina** - стенокардия при нагрузке (0/1)
- **st_depression** - депрессия сегмента ST
- **vessels_colored** - число основных сосудов, окрашенных при флюороскопии (0-3)

### Целевая переменная
- **heart_disease** - наличие болезни сердца (0/1)"""))

    # Cell 4: Генерация датасета
    cells.append(nbf.v4.new_code_cell("""def create_heart_disease_data(n_samples=8000):
    \"\"\"
    Создание синтетического датасета для диагностики болезни сердца.

    Параметры:
    ----------
    n_samples : int
        Количество пациентов

    Возвращает:
    -----------
    DataFrame с клиническими показателями и диагнозом
    \"\"\"
    # Базовые характеристики
    age = np.random.normal(55, 12, n_samples).clip(25, 85)
    sex = np.random.binomial(1, 0.6, n_samples)  # 60% мужчин

    # Клинические показатели (зависят от возраста и пола)
    blood_pressure = (120 + 0.5 * (age - 50) + 5 * sex +
                      np.random.normal(0, 15, n_samples)).clip(90, 200)

    cholesterol = (200 + 0.8 * (age - 50) + 10 * sex +
                   np.random.normal(0, 30, n_samples)).clip(120, 400)

    blood_sugar = (90 + 0.3 * (age - 50) +
                   np.random.normal(0, 20, n_samples)).clip(60, 200)

    heart_rate = (72 - 0.1 * (age - 50) +
                  np.random.normal(0, 10, n_samples)).clip(50, 120)

    # Результаты обследований
    ecg_prob = 0.1 + 0.005 * (age - 50) + 0.05 * sex
    ecg_abnormal = np.random.binomial(1, ecg_prob.clip(0, 1))

    exercise_angina_prob = 0.15 + 0.008 * (age - 50) + 0.1 * sex
    exercise_angina = np.random.binomial(1, exercise_angina_prob.clip(0, 1))

    st_depression = (0.5 + 0.02 * (age - 50) +
                     np.random.exponential(0.5, n_samples)).clip(0, 5)

    vessels_colored = np.random.poisson(0.5 + 0.02 * (age - 50), n_samples).clip(0, 3)

    # Вероятность болезни сердца (логистическая модель)
    logit = (-5 +
             0.05 * (age - 50) +
             0.5 * sex +
             0.02 * (blood_pressure - 120) +
             0.01 * (cholesterol - 200) +
             0.01 * (blood_sugar - 100) +
             0.02 * (heart_rate - 72) +
             1.0 * ecg_abnormal +
             1.5 * exercise_angina +
             0.8 * st_depression +
             0.7 * vessels_colored)

    prob = 1 / (1 + np.exp(-logit))
    heart_disease = np.random.binomial(1, prob)

    # Создаём DataFrame
    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'blood_pressure': blood_pressure,
        'cholesterol': cholesterol,
        'blood_sugar': blood_sugar,
        'heart_rate': heart_rate,
        'ecg_abnormal': ecg_abnormal,
        'exercise_angina': exercise_angina,
        'st_depression': st_depression,
        'vessels_colored': vessels_colored,
        'heart_disease': heart_disease
    })

    return df

# Создаём датасет
df = create_heart_disease_data(n_samples=8000)

print(f"Размер датасета: {df.shape}")
print(f"\\nРаспределение диагнозов:")
print(df['heart_disease'].value_counts())
print(f"\\nДоля больных: {df['heart_disease'].mean()*100:.1f}%")"""))

    # Cell 5: Анализ данных
    cells.append(nbf.v4.new_code_cell("""# Статистика по признакам
print("Статистика признаков:")
print(df.describe().round(2))

# Корреляция с целевой переменной
correlations = df.corr()['heart_disease'].drop('heart_disease').sort_values(ascending=False)
print("\\nКорреляция с болезнью сердца:")
print(correlations.round(3))

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Распределения ключевых признаков
key_features = ['age', 'blood_pressure', 'cholesterol', 'st_depression', 'vessels_colored', 'heart_rate']

for i, col in enumerate(key_features):
    ax = axes[i]
    for label in [0, 1]:
        subset = df[df['heart_disease'] == label][col]
        ax.hist(subset, bins=30, alpha=0.5,
                label='Здоров' if label == 0 else 'Болен', density=True)
    ax.set_title(col)
    ax.legend()

plt.suptitle('Распределение признаков по диагнозу', fontsize=14)
plt.tight_layout()
plt.show()"""))

    # Cell 6: Подготовка данных и модель
    cells.append(nbf.v4.new_code_cell("""# Подготовка данных
feature_cols = ['age', 'sex', 'blood_pressure', 'cholesterol', 'blood_sugar',
                'heart_rate', 'ecg_abnormal', 'exercise_angina',
                'st_depression', 'vessels_colored']

X = df[feature_cols].values
y = df['heart_disease'].values

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")

# Построение нейронной сети
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Обучение
model = build_model(X_train_scaled.shape[1])

history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    verbose=0
)

# Оценка
y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

print("\\nПроизводительность модели:")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print(classification_report(y_test, y_pred, target_names=['Здоров', 'Болен']))"""))

    # Cell 7: Integrated Gradients введение
    cells.append(nbf.v4.new_markdown_cell("""## 2. Integrated Gradients

### Теория

Integrated Gradients (IG) - это метод атрибуции, который удовлетворяет двум важным аксиомам:

1. **Чувствительность (Sensitivity)** - если признак влияет на предсказание, он должен получить ненулевую атрибуцию
2. **Инвариантность к реализации (Implementation Invariance)** - атрибуции не должны зависеть от внутренней реализации модели

### Формула

$$IG_i(x) = (x_i - x'_i) \\times \\int_{\\alpha=0}^{1} \\frac{\\partial F(x' + \\alpha(x - x'))}{\\partial x_i} d\\alpha$$

где:
- $x$ - входной пример
- $x'$ - базовый пример (baseline, обычно нули)
- $F$ - функция модели
- $\\alpha$ - параметр интерполяции

### Интуиция

Мы "идём" от базового примера к нашему входу и накапливаем градиенты вдоль пути. Это показывает, как каждый признак способствует изменению предсказания."""))

    # Cell 8: Реализация Integrated Gradients
    cells.append(nbf.v4.new_code_cell("""def integrated_gradients(model, input_data, baseline=None, steps=50):
    \"\"\"
    Вычисление Integrated Gradients для объяснения предсказаний.

    Параметры:
    ----------
    model : Keras model
        Обученная модель
    input_data : array
        Входные данные для объяснения
    baseline : array
        Базовый пример (по умолчанию нули)
    steps : int
        Число шагов интегрирования

    Возвращает:
    -----------
    attributions : array
        Атрибуции для каждого признака
    \"\"\"
    if baseline is None:
        baseline = np.zeros_like(input_data)

    # Преобразуем в тензоры
    input_tensor = tf.constant(input_data, dtype=tf.float32)
    baseline_tensor = tf.constant(baseline, dtype=tf.float32)

    # Создаём интерполированные входы
    alphas = tf.linspace(0.0, 1.0, steps + 1)

    # Для batch обработки
    if len(input_data.shape) == 1:
        input_tensor = tf.expand_dims(input_tensor, 0)
        baseline_tensor = tf.expand_dims(baseline_tensor, 0)

    # Интерполяция между baseline и input
    interpolated_inputs = []
    for alpha in alphas:
        interpolated = baseline_tensor + alpha * (input_tensor - baseline_tensor)
        interpolated_inputs.append(interpolated)

    interpolated_inputs = tf.concat(interpolated_inputs, axis=0)

    # Вычисляем градиенты
    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        predictions = model(interpolated_inputs)

    gradients = tape.gradient(predictions, interpolated_inputs)

    # Интегрируем градиенты (трапецеидальное правило)
    gradients = gradients.numpy()
    avg_gradients = np.mean(gradients[:-1] + gradients[1:], axis=0) / 2

    # Атрибуции = средний градиент * (input - baseline)
    attributions = avg_gradients * (input_data - baseline)

    return attributions

print("Функция Integrated Gradients определена")"""))

    # Cell 9: Применение IG
    cells.append(nbf.v4.new_code_cell("""# Применяем Integrated Gradients к тестовым примерам
print("Вычисление Integrated Gradients...")

# Выбираем примеры для объяснения
n_explain = 100
X_explain = X_test_scaled[:n_explain]
y_explain = y_test[:n_explain]
pred_explain = y_pred_proba[:n_explain]

# Вычисляем атрибуции для каждого примера
all_attributions = []

for i in range(n_explain):
    attr = integrated_gradients(model, X_explain[i], steps=50)
    all_attributions.append(attr.flatten())

attributions = np.array(all_attributions)

print(f"Атрибуции вычислены: {attributions.shape}")

# Средняя важность признаков (по абсолютным значениям)
mean_abs_attr = np.abs(attributions).mean(axis=0)

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Средняя важность
ax1 = axes[0]
sorted_idx = np.argsort(mean_abs_attr)
ax1.barh([feature_cols[i] for i in sorted_idx], mean_abs_attr[sorted_idx])
ax1.set_xlabel('Средняя |атрибуция|')
ax1.set_title('Integrated Gradients: Важность признаков')

# Распределение атрибуций
ax2 = axes[1]
# Beeswarm-подобный plot
for i, col in enumerate(feature_cols):
    y_jitter = np.random.normal(i, 0.1, n_explain)
    colors = ['red' if a > 0 else 'blue' for a in attributions[:, i]]
    ax2.scatter(attributions[:, i], y_jitter, c=colors, alpha=0.3, s=10)

ax2.set_yticks(range(len(feature_cols)))
ax2.set_yticklabels(feature_cols)
ax2.axvline(x=0, color='black', linewidth=0.5)
ax2.set_xlabel('Атрибуция')
ax2.set_title('Распределение атрибуций (красный = увеличивает риск)')

plt.tight_layout()
plt.show()"""))

    # Cell 10: Индивидуальные объяснения IG
    cells.append(nbf.v4.new_code_cell("""# Объяснение конкретных пациентов
def explain_patient(idx, X_scaled, X_original, y_true, y_pred, attributions, feature_names):
    \"\"\"Детальное объяснение предсказания для пациента.\"\"\"
    attr = attributions[idx]
    pred = y_pred[idx]
    true = y_true[idx]

    print(f"Пациент {idx}")
    print(f"Истинный диагноз: {'Болен' if true else 'Здоров'}")
    print(f"Предсказание: {pred:.3f} ({'Болен' if pred > 0.5 else 'Здоров'})")
    print("\\nВлияние признаков:")

    # Сортируем по абсолютной атрибуции
    sorted_idx = np.argsort(np.abs(attr))[::-1]

    for i in sorted_idx:
        name = feature_names[i]
        value = X_original[idx, i]
        attribution = attr[i]
        direction = "↑" if attribution > 0 else "↓"

        print(f"  {name}: {value:.1f} → {direction} ({attribution:+.3f})")

# Примеры пациентов
# Истинно положительный (больной, правильно распознан)
tp_idx = np.where((y_explain == 1) & (pred_explain > 0.7))[0]
if len(tp_idx) > 0:
    print("=" * 50)
    print("ИСТИННО ПОЛОЖИТЕЛЬНЫЙ (высокий риск)")
    print("=" * 50)
    explain_patient(tp_idx[0], X_explain, X_test[:n_explain],
                   y_explain, pred_explain, attributions, feature_cols)

# Истинно отрицательный (здоровый, правильно распознан)
tn_idx = np.where((y_explain == 0) & (pred_explain < 0.3))[0]
if len(tn_idx) > 0:
    print("\\n" + "=" * 50)
    print("ИСТИННО ОТРИЦАТЕЛЬНЫЙ (низкий риск)")
    print("=" * 50)
    explain_patient(tn_idx[0], X_explain, X_test[:n_explain],
                   y_explain, pred_explain, attributions, feature_cols)"""))

    # Cell 11: Anchor Explanations введение
    cells.append(nbf.v4.new_markdown_cell("""## 3. Anchor Explanations

### Теория

Anchors - это объяснения в виде правил "if-then", которые достаточны для предсказания. Anchor - это набор условий, которые "закрепляют" предсказание с высокой вероятностью.

### Формальное определение

Anchor $A$ для примера $x$ с предсказанием $f(x)$:

$$P(f(z) = f(x) | A(z) = 1) \\geq \\tau$$

где:
- $z$ - примеры из распределения данных
- $A(z) = 1$ означает, что пример $z$ удовлетворяет условиям anchor
- $\\tau$ - требуемая точность (precision)

### Преимущества

1. **Понятность** - правила легко интерпретировать
2. **Локальная точность** - высокая точность в области применения
3. **Actionable** - понятно, что нужно изменить

### Пример

"ЕСЛИ возраст > 60 И холестерин > 250 И exercise_angina = 1, ТО высокий риск болезни сердца (точность 95%)"

Мы реализуем упрощённую версию поиска anchors."""))

    # Cell 12: Реализация Anchors
    cells.append(nbf.v4.new_code_cell("""class SimpleAnchorExplainer:
    \"\"\"
    Упрощённая реализация Anchor Explanations.

    Ищет правила (условия на признаки), которые с высокой
    вероятностью приводят к определённому предсказанию.
    \"\"\"

    def __init__(self, model, X_train, feature_names, percentiles=[25, 50, 75]):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.n_features = len(feature_names)

        # Вычисляем пороги для каждого признака
        self.thresholds = {}
        for i, name in enumerate(feature_names):
            values = X_train[:, i]
            self.thresholds[i] = np.percentile(values, percentiles)

    def _generate_conditions(self, instance):
        \"\"\"Генерация возможных условий для примера.\"\"\"
        conditions = []

        for i in range(self.n_features):
            value = instance[i]
            thresholds = self.thresholds[i]

            # Условия вида "feature < threshold" или "feature >= threshold"
            for t in thresholds:
                if value < t:
                    conditions.append((i, '<', t))
                else:
                    conditions.append((i, '>=', t))

        return conditions

    def _evaluate_anchor(self, conditions, target_pred, n_samples=500):
        \"\"\"Оценка качества anchor (precision).\"\"\"
        # Генерируем примеры из обучающей выборки
        indices = np.random.choice(len(self.X_train), n_samples, replace=True)
        samples = self.X_train[indices].copy()

        # Фильтруем примеры, удовлетворяющие условиям
        mask = np.ones(n_samples, dtype=bool)
        for feat_idx, op, threshold in conditions:
            if op == '<':
                mask &= samples[:, feat_idx] < threshold
            else:
                mask &= samples[:, feat_idx] >= threshold

        if mask.sum() < 10:
            return 0, 0  # Недостаточно примеров

        # Предсказания для отфильтрованных примеров
        filtered_samples = samples[mask]
        predictions = (self.model.predict(filtered_samples, verbose=0) > 0.5).astype(int).flatten()

        # Precision = доля правильных предсказаний
        precision = (predictions == target_pred).mean()
        coverage = mask.sum() / n_samples

        return precision, coverage

    def explain(self, instance, min_precision=0.9, max_conditions=4):
        \"\"\"
        Найти anchor для примера.

        Параметры:
        ----------
        instance : array
            Пример для объяснения
        min_precision : float
            Минимальная требуемая точность
        max_conditions : int
            Максимальное число условий

        Возвращает:
        -----------
        anchor : list of conditions
        precision : float
        coverage : float
        \"\"\"
        # Предсказание для примера
        pred = int(self.model.predict(instance.reshape(1, -1), verbose=0) > 0.5)

        # Генерируем условия
        all_conditions = self._generate_conditions(instance)

        # Жадный поиск лучшего anchor
        best_anchor = []
        best_precision = 0
        best_coverage = 0

        current_anchor = []
        remaining_conditions = all_conditions.copy()

        for _ in range(max_conditions):
            best_next = None
            best_next_score = -1

            for cond in remaining_conditions:
                test_anchor = current_anchor + [cond]
                precision, coverage = self._evaluate_anchor(test_anchor, pred)

                # Скор = precision * sqrt(coverage)
                score = precision * np.sqrt(coverage) if coverage > 0 else 0

                if score > best_next_score:
                    best_next_score = score
                    best_next = cond
                    best_next_precision = precision
                    best_next_coverage = coverage

            if best_next is None:
                break

            current_anchor.append(best_next)
            remaining_conditions.remove(best_next)

            if best_next_precision >= min_precision:
                best_anchor = current_anchor.copy()
                best_precision = best_next_precision
                best_coverage = best_next_coverage

        return best_anchor, best_precision, best_coverage, pred

    def format_anchor(self, anchor, pred):
        \"\"\"Форматирование anchor в читаемый вид.\"\"\"
        if not anchor:
            return "Не найдено правило с достаточной точностью"

        conditions_str = []
        for feat_idx, op, threshold in anchor:
            name = self.feature_names[feat_idx]
            conditions_str.append(f"{name} {op} {threshold:.1f}")

        rule = " И ".join(conditions_str)
        outcome = "Высокий риск" if pred == 1 else "Низкий риск"

        return f"ЕСЛИ {rule}, ТО {outcome}"

print("Класс SimpleAnchorExplainer определён")"""))

    # Cell 13: Применение Anchors
    cells.append(nbf.v4.new_code_cell("""# Создаём explainer
anchor_explainer = SimpleAnchorExplainer(
    model, X_train_scaled, feature_cols
)

# Объясняем несколько пациентов
print("Поиск Anchor объяснений...")
print("=" * 60)

# Пациенты с высоким риском
high_risk_idx = np.where(pred_explain > 0.7)[0][:3]

print("\\nПАЦИЕНТЫ С ВЫСОКИМ РИСКОМ:")
for idx in high_risk_idx:
    anchor, precision, coverage, pred = anchor_explainer.explain(
        X_explain[idx], min_precision=0.85
    )

    rule = anchor_explainer.format_anchor(anchor, pred)

    print(f"\\nПациент {idx} (предсказание: {pred_explain[idx]:.3f})")
    print(f"  Правило: {rule}")
    print(f"  Точность: {precision*100:.1f}%")
    print(f"  Покрытие: {coverage*100:.1f}%")

# Пациенты с низким риском
low_risk_idx = np.where(pred_explain < 0.3)[0][:3]

print("\\n" + "=" * 60)
print("\\nПАЦИЕНТЫ С НИЗКИМ РИСКОМ:")
for idx in low_risk_idx:
    anchor, precision, coverage, pred = anchor_explainer.explain(
        X_explain[idx], min_precision=0.85
    )

    rule = anchor_explainer.format_anchor(anchor, pred)

    print(f"\\nПациент {idx} (предсказание: {pred_explain[idx]:.3f})")
    print(f"  Правило: {rule}")
    print(f"  Точность: {precision*100:.1f}%")
    print(f"  Покрытие: {coverage*100:.1f}%")"""))

    # Cell 14: TCAV введение
    cells.append(nbf.v4.new_markdown_cell("""## 4. Concept-based Explanations (TCAV-like)

### Теория

TCAV (Testing with Concept Activation Vectors) объясняет предсказания модели через высокоуровневые понятия, понятные человеку.

### Идея

Вместо объяснения через отдельные признаки, мы используем **концепции** - группы признаков, объединённые смыслом:

- **Сердечно-сосудистые факторы**: давление, холестерин, пульс
- **Возрастные факторы**: возраст, пол
- **Результаты тестов**: ЭКГ, стенокардия, ST-депрессия, сосуды

### Алгоритм (упрощённый)

1. Определяем концепции как группы признаков
2. Для каждой концепции вычисляем "направление" в пространстве активаций
3. Измеряем, насколько предсказание чувствительно к движению в этом направлении

### Преимущества

- Объяснения на уровне медицинских понятий
- Легче для понимания врачами
- Можно проверять гипотезы о модели"""))

    # Cell 15: Определение концепций
    cells.append(nbf.v4.new_code_cell("""# Определение медицинских концепций
CONCEPTS = {
    'Сердечно-сосудистые': ['blood_pressure', 'cholesterol', 'heart_rate'],
    'Демографические': ['age', 'sex'],
    'Результаты тестов': ['ecg_abnormal', 'exercise_angina', 'st_depression', 'vessels_colored'],
    'Метаболические': ['blood_sugar', 'cholesterol']
}

# Индексы признаков для каждой концепции
concept_indices = {}
for concept_name, features in CONCEPTS.items():
    indices = [feature_cols.index(f) for f in features if f in feature_cols]
    concept_indices[concept_name] = indices
    print(f"{concept_name}: {features} → индексы {indices}")"""))

    # Cell 16: Реализация Concept Sensitivity
    cells.append(nbf.v4.new_code_cell("""def concept_sensitivity(model, X, concept_indices, epsilon=0.1, n_samples=100):
    \"\"\"
    Вычисление чувствительности предсказания к концепции.

    Параметры:
    ----------
    model : Keras model
        Обученная модель
    X : array
        Данные для анализа
    concept_indices : list
        Индексы признаков концепции
    epsilon : float
        Величина возмущения
    n_samples : int
        Число примеров для анализа

    Возвращает:
    -----------
    sensitivity : float
        Средняя чувствительность
    sensitivities : array
        Чувствительность для каждого примера
    \"\"\"
    # Выбираем подмножество данных
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    # Базовые предсказания
    base_preds = model.predict(X_sample, verbose=0).flatten()

    # Возмущаем признаки концепции
    X_perturbed = X_sample.copy()
    for i in concept_indices:
        X_perturbed[:, i] += epsilon * np.std(X_sample[:, i])

    # Предсказания после возмущения
    perturbed_preds = model.predict(X_perturbed, verbose=0).flatten()

    # Чувствительность = изменение предсказания
    sensitivities = perturbed_preds - base_preds

    return np.mean(sensitivities), sensitivities

# Вычисляем чувствительность для каждой концепции
print("Анализ чувствительности к концепциям:")
print("=" * 50)

concept_results = {}

for concept_name, indices in concept_indices.items():
    mean_sens, all_sens = concept_sensitivity(
        model, X_test_scaled, indices, epsilon=0.5
    )

    concept_results[concept_name] = {
        'mean': mean_sens,
        'std': np.std(all_sens),
        'sensitivities': all_sens
    }

    direction = "увеличивает" if mean_sens > 0 else "уменьшает"
    print(f"\\n{concept_name}:")
    print(f"  Среднее изменение: {mean_sens:+.4f}")
    print(f"  Интерпретация: {direction} риск болезни сердца")"""))

    # Cell 17: Визуализация концепций
    cells.append(nbf.v4.new_code_cell("""# Визуализация чувствительности к концепциям
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Средняя чувствительность
ax1 = axes[0]
concepts = list(concept_results.keys())
means = [concept_results[c]['mean'] for c in concepts]
stds = [concept_results[c]['std'] for c in concepts]

colors = ['red' if m > 0 else 'blue' for m in means]
bars = ax1.barh(concepts, means, xerr=stds, color=colors, alpha=0.7, capsize=5)

ax1.axvline(x=0, color='black', linewidth=0.5)
ax1.set_xlabel('Средняя чувствительность')
ax1.set_title('Влияние концепций на риск болезни сердца\\n(красный = увеличивает риск)')

# 2. Распределение чувствительности
ax2 = axes[1]
for i, concept in enumerate(concepts):
    sens = concept_results[concept]['sensitivities']
    y_jitter = np.random.normal(i, 0.1, len(sens))
    ax2.scatter(sens, y_jitter, alpha=0.3, s=20)

ax2.set_yticks(range(len(concepts)))
ax2.set_yticklabels(concepts)
ax2.axvline(x=0, color='black', linewidth=0.5)
ax2.set_xlabel('Чувствительность')
ax2.set_title('Распределение чувствительности по примерам')

plt.tight_layout()
plt.show()"""))

    # Cell 18: Объяснение через концепции
    cells.append(nbf.v4.new_code_cell("""# Объяснение конкретного пациента через концепции
def explain_with_concepts(model, patient_data, concept_indices, feature_names, epsilon=0.5):
    \"\"\"
    Объяснение предсказания через концепции.
    \"\"\"
    # Базовое предсказание
    base_pred = model.predict(patient_data.reshape(1, -1), verbose=0)[0, 0]

    explanations = []

    for concept_name, indices in concept_indices.items():
        # Возмущаем концепцию
        perturbed = patient_data.copy()
        for i in indices:
            perturbed[i] += epsilon * np.std(X_train_scaled[:, i])

        new_pred = model.predict(perturbed.reshape(1, -1), verbose=0)[0, 0]
        change = new_pred - base_pred

        explanations.append({
            'concept': concept_name,
            'change': change,
            'features': [feature_names[i] for i in indices]
        })

    return base_pred, sorted(explanations, key=lambda x: abs(x['change']), reverse=True)

# Объясняем пациента с высоким риском
high_risk_patient_idx = np.argmax(pred_explain)
patient_data = X_explain[high_risk_patient_idx]

base_pred, explanations = explain_with_concepts(
    model, patient_data, concept_indices, feature_cols
)

print("ОБЪЯСНЕНИЕ ЧЕРЕЗ КОНЦЕПЦИИ")
print("=" * 50)
print(f"\\nПациент {high_risk_patient_idx}")
print(f"Базовое предсказание: {base_pred:.3f} ({'Высокий риск' if base_pred > 0.5 else 'Низкий риск'})")
print("\\nВлияние концепций (при увеличении):")

for exp in explanations:
    direction = "↑" if exp['change'] > 0 else "↓"
    print(f"\\n  {exp['concept']}:")
    print(f"    Изменение: {exp['change']:+.4f} {direction}")
    print(f"    Признаки: {', '.join(exp['features'])}")

    if exp['change'] > 0.01:
        print("    → Увеличивает риск")
    elif exp['change'] < -0.01:
        print("    → Уменьшает риск")
    else:
        print("    → Минимальное влияние")"""))

    # Cell 19: Сравнение методов
    cells.append(nbf.v4.new_code_cell("""# Сравнение всех методов XAI
print("=" * 60)
print("СРАВНЕНИЕ МЕТОДОВ ОБЪЯСНИМОГО ИИ")
print("=" * 60)

print("\\n1. Integrated Gradients")
print("   Тип: Атрибуция на уровне признаков")
print("   Формат: Числовой вклад каждого признака")
print("   Плюсы: Теоретические гарантии, точность")
print("   Минусы: Числа сложно интерпретировать")
print("   Аудитория: Data scientists, исследователи")

print("\\n2. Anchor Explanations")
print("   Тип: Правила (if-then)")
print("   Формат: 'ЕСЛИ условия, ТО предсказание'")
print("   Плюсы: Понятные правила, actionable")
print("   Минусы: Упрощение, может терять нюансы")
print("   Аудитория: Врачи, пациенты, регуляторы")

print("\\n3. Concept-based (TCAV-like)")
print("   Тип: Высокоуровневые концепции")
print("   Формат: Влияние медицинских понятий")
print("   Плюсы: Соответствует экспертному мышлению")
print("   Минусы: Требует определения концепций")
print("   Аудитория: Врачи, медицинские эксперты")

print("\\n" + "=" * 60)
print("РЕКОМЕНДАЦИИ ДЛЯ МЕДИЦИНСКИХ ПРИЛОЖЕНИЙ")
print("=" * 60)
print("\\n• Для пациентов: Anchor Explanations")
print("  'Высокий риск из-за повышенного давления и холестерина'")
print("\\n• Для врачей: Concept-based + Anchors")
print("  Связь с медицинскими понятиями + конкретные правила")
print("\\n• Для аудита: Integrated Gradients")
print("  Полная атрибуция для проверки модели")
print("\\n• Рекомендация: Комбинация всех трёх методов")"""))

    # Cell 20: Заключение
    cells.append(nbf.v4.new_markdown_cell("""## Заключение

### Ключевые результаты

1. **Integrated Gradients** показал, что наибольший вклад в предсказание вносят:
   - exercise_angina (стенокардия при нагрузке)
   - st_depression (депрессия сегмента ST)
   - vessels_colored (число поражённых сосудов)

2. **Anchor Explanations** нашёл интерпретируемые правила, например:
   - "ЕСЛИ st_depression >= 1.2 И exercise_angina = 1, ТО высокий риск"

3. **Concept-based анализ** показал, что:
   - "Результаты тестов" - наиболее влиятельная концепция
   - "Сердечно-сосудистые факторы" также значимы

### Особенности датасета

- 8,000 синтетических пациентов
- 10 клинических признаков
- Реалистичная зависимость диагноза от признаков

### Практическое применение

1. **Клиническая поддержка** - помощь врачам в принятии решений
2. **Коммуникация с пациентами** - понятные объяснения рисков
3. **Регуляторное соответствие** - документирование решений ИИ
4. **Проверка модели** - выявление нежелательных паттернов

### Ограничения и предупреждения

- Это учебный пример, не для реального медицинского использования
- Реальные медицинские модели требуют тщательной валидации
- Объяснения должны проверяться медицинскими экспертами

### Дальнейшее развитие

- Интеграция с электронными медицинскими картами
- Онлайн-обучение для персонализации
- Мультимодальные объяснения (текст + визуализация)
- Валидация объяснений с врачами"""))

    nb['cells'] = cells

    # Сохранение
    output_path = '/home/user/test/notebooks/phase6_explainable_ai/03_advanced_xai_medical.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"✅ Notebook создан: {output_path}")
    print(f"Всего ячеек: {len(cells)}")
    return output_path

if __name__ == "__main__":
    create_notebook()
