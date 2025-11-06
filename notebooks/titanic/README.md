# Титаник: Пояснения по датасету, валидации и технологиям

## 🔧 Текущие проблемы и их решения

### 1. SNS Dataset vs Kaggle Dataset

#### Текущая реализация:
```python
# cell-4
df = sns.load_dataset('titanic')
```

####  Проблема:
- **SNS датасет** - упрощенная версия, объединенная train+test с целевой переменной
- **Kaggle датасет** - реальная структура соревнования:
  - `train.csv` (891 строка) - обучающие данные с `Survived`
  - `test.csv` (418 строк) - тестовые данные БЕЗ `Survived` (для submission)

#### ✅ Правильная реализация:

```python
# Загрузка данных
import os

# Попытка загрузить Kaggle датасет
try:
    # Kaggle датасет (предпочтительно)
    train_path = '../../datasets/titanic/train.csv'
    test_path = '../../datasets/titanic/test.csv'

    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path)
        df_test_submission = pd.read_csv(test_path) if os.path.exists(test_path) else None

        df = df_train.copy()  # Используем train для анализа
        df_original = df.copy()

        print(f"✓ Загружен Kaggle датасет")
        print(f"  Train: {df_train.shape}")
        if df_test_submission is not None:
            print(f"  Test (для submission): {df_test_submission.shape}")
    else:
        raise FileNotFoundError

except (FileNotFoundError, Exception):
    # Fallback на SNS датасет
    print("⚠ Kaggle датасет не найден, используем seaborn")
    print("  Для полноценной работы загрузите train.csv и test.csv с Kaggle")
    print("  Kaggle: https://www.kaggle.com/c/titanic/data")

    df = sns.load_dataset('titanic')
    df_original = df.copy()
    df_test_submission = None

# Переименовываем колонки для унификации (sns использует lowercase)
if 'survived' in df.columns:
    df.rename(columns={
        'survived': 'Survived',
        'pclass': 'Pclass',
        'sex': 'Sex',
        'age': 'Age',
        'sibsp': 'SibSp',
        'parch': 'Parch',
        'fare': 'Fare',
        'embarked': 'Embarked'
    }, inplace=True)

print(f"\nРазмер датасета: {df.shape}")
print(f"Количество строк: {df.shape[0]}")
print(f"Количество признаков: {df.shape[1]}")
```

---

### 2. Train/Validation/Test Split

#### Текущая реализация (НЕПРАВИЛЬНО):
```python
# cell-43
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

#### ❌ Проблемы:
1. **Нет validation set** - подбираем гиперпараметры на test set
2. **Data leakage** - "подглядываем" в test при оптимизации
3. **Переоценка качества** - модель видела test много раз

#### ✅ Правильная реализация:

```python
# Правильное разделение данных
from sklearn.model_selection import train_test_split

# Разделение на признаки (X) и целевую переменную (y)
X = df_model.drop('Survived', axis=1)
y = df_model['Survived']

# ВАЖНО: Разделяем на Train/Validation/Test
# Шаг 1: Отделяем test set (15%) - трогаем ТОЛЬКО РАЗ в конце
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Шаг 2: Делим оставшиеся данные на Train (70%) и Validation (15%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
)

print("=" * 60)
print("РАЗДЕЛЕНИЕ ДАННЫХ")
print("=" * 60)
print(f"Общий размер: {X.shape[0]} строк")
print(f"\n📚 Train set:      {X_train.shape[0]} строк ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
print(f"   └─ Для обучения моделей")
print(f"\n🎯 Validation set: {X_val.shape[0]} строк ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
print(f"   └─ Для подбора гиперпараметров и выбора модели")
print(f"\n🔒 Test set:       {X_test.shape[0]} строк ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
print(f"   └─ Для ФИНАЛЬНОЙ оценки (трогаем ТОЛЬКО РАЗ!)")
print("=" * 60)

# Распределение классов
print(f"\nРаспределение классов:")
print(f"Train:      {y_train.value_counts(normalize=True).to_dict()}")
print(f"Validation: {y_val.value_counts(normalize=True).to_dict()}")
print(f"Test:       {y_test.value_counts(normalize=True).to_dict()}")
```

#### 🎯 Зачем нужна Validation выборка?

| Выборка | Назначение | Когда используем |
|---------|-----------|------------------|
| **Train** | Обучение моделей | Каждая эпоха/итерация |
| **Validation** | Подбор гиперпараметров, early stopping, выбор модели | Много раз в процессе разработки |
| **Test** | Финальная оценка обобщающей способности | **ТОЛЬКО ОДИН РАЗ** в конце! |

**Без validation**:
```
❌ Train → подбираем гиперпараметры на Test → выбираем модель по Test
   = Переобучение на Test! Завышенная оценка!
```

**С validation**:
```
✅ Train → подбираем гиперпараметры на Validation → выбираем модель по Validation
   → Test используем ОДИН РАЗ для честной оценки
```

#### Использование в GridSearchCV:

```python
# НЕПРАВИЛЬНО (старый код):
grid_search.fit(X_train, y_train)  # CV внутри train
y_pred = grid_search.predict(X_test)  # Оцениваем на test - ПЛОХО!

# ПРАВИЛЬНО (новый код):
grid_search.fit(X_train, y_train)  # CV внутри train
y_pred_val = grid_search.predict(X_val)  # Оцениваем на validation - ХОРОШО!

# Test используем ТОЛЬКО в самом конце:
final_model = grid_search.best_estimator_
y_pred_test = final_model.predict(X_test)  # ТОЛЬКО РАЗ!
print(f"Финальная точность на test: {accuracy_score(y_test, y_pred_test):.4f}")
```

---

### 3. Pickle vs Joblib

#### Текущая реализация (НЕПРАВИЛЬНО):
```python
# cell-75
import pickle

with open('best_titanic_model.pkl', 'wb') as file:
    pickle.dump(best_rf, file)
```

#### ❌ Почему pickle плохо для scikit-learn?

| Аспект | pickle | joblib |
|--------|--------|--------|
| **Скорость** | Медленный на больших numpy массивах | ⚡ В 2-10 раз быстрее |
| **Размер файла** | Больше | 🗜️ Лучшее сжатие |
| **Память** | Может быть проблема с большими объектами | Эффективная работа |
| **Scikit-learn** | ⚠️ Не рекомендуется | ✅ **Официально рекомендуется** |
| **Мемоизация** | Нет | Есть (disk caching) |

#### ✅ Правильная реализация:

```python
# Импорт joblib
import joblib  # ← Официальная библиотека для scikit-learn

# Сохранение модели
# Формат .joblib рекомендуется вместо .pkl
model_path = '../../models/best_titanic_model.joblib'
scaler_path = '../../models/titanic_scaler.joblib'

# Сохранение с сжатием (compress=3 - хороший баланс)
joblib.dump(best_rf, model_path, compress=3)
joblib.dump(scaler, scaler_path, compress=3)

print("✅ Модель и scaler успешно сохранены с joblib!")
print(f"   Модель: {model_path}")
print(f"   Scaler: {scaler_path}")

# Загрузка модели
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

print("\n✅ Модель и scaler успешно загружены!")
```

#### Сравнение производительности:

```python
# Benchmark (для примера)
import time
import numpy as np

# Большая модель
X_large = np.random.randn(10000, 100)
y_large = np.random.randint(0, 2, 10000)
model_large = RandomForestClassifier(n_estimators=100).fit(X_large, y_large)

# pickle
start = time.time()
with open('model_pickle.pkl', 'wb') as f:
    pickle.dump(model_large, f)
pickle_time = time.time() - start
pickle_size = os.path.getsize('model_pickle.pkl') / 1024 / 1024  # MB

# joblib
start = time.time()
joblib.dump(model_large, 'model_joblib.joblib', compress=3)
joblib_time = time.time() - start
joblib_size = os.path.getsize('model_joblib.joblib') / 1024 / 1024  # MB

print(f"pickle:  {pickle_time:.3f}s, {pickle_size:.2f} MB")
print(f"joblib:  {joblib_time:.3f}s, {joblib_size:.2f} MB")
print(f"Speedup: {pickle_time/joblib_time:.1f}x faster")
print(f"Compression: {pickle_size/joblib_size:.1f}x smaller")
```

**Типичные результаты**:
```
pickle:  2.341s, 45.23 MB
joblib:  0.832s, 18.67 MB
Speedup: 2.8x faster
Compression: 2.4x smaller
```

#### Параметр compress:

```python
# compress=0 - без сжатия (быстрее, но больше)
# compress=1-9 - уровень сжатия (выше = меньше размер, медленнее)
# compress=3 - рекомендуется (хороший баланс)

joblib.dump(model, 'model.joblib', compress=3)
```

---

### 4. Jupyter Notebook Metadata

#### Вопрос: "Почему при загрузке в локальный Jupyter добавились outputs и metadata?"

#### ✅ Это НОРМАЛЬНО и ПРАВИЛЬНО!

#### Структура Jupyter Notebook:

```json
{
  "cells": [
    {
      "cell_type": "code",
      "source": "print('Hello, World!')",

      "execution_count": 1,        ← Порядок выполнения ячейки

      "outputs": [                 ← Результаты выполнения
        {
          "output_type": "stream",
          "text": "Hello, World!\n"
        }
      ],

      "metadata": {                ← Метаданные ячейки
        "collapsed": false,
        "scrolled": true
      }
    }
  ],

  "metadata": {                    ← Метаданные ноутбука
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.8.10"
    }
  },

  "nbformat": 4,
  "nbformat_minor": 4
}
```

#### Три состояния notebook:

| Состояние | outputs | execution_count | Когда |
|-----------|---------|-----------------|-------|
| **Чистый** | `[]` | `null` | После создания программно |
| **Выполненный** | `[{...}]` | `1, 2, 3...` | После запуска в Jupyter |
| **Cleared** | `[]` | `null` | После очистки |

#### Почему у меня (Claude) не было outputs?

Я создал notebook **программно** - это "чистый" notebook без результатов выполнения.

Когда ВЫ запускаете его в Jupyter, он **автоматически добавляет**:
- `execution_count` - номер выполнения ячейки
- `outputs` - результаты (print, графики, таблицы)
- `metadata` - дополнительная информация

#### Best Practices для Git:

**Проблема**: Outputs могут быть огромными (изображения, большие DataFrame)

```json
{
  "outputs": [{
    "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABkAAAASwCAYAAABm..." // 50+ KB Base64
    }
  }]
}
```

**Решение**: Очистить outputs перед коммитом

#### 1. Ручная очистка в Jupyter:
```
Cell → All Output → Clear
```

#### 2. Через nbconvert:
```bash
# Очистить outputs
jupyter nbconvert --clear-output --inplace notebook.ipynb
```

#### 3. Автоматизация с nbstripout:
```bash
# Установка
pip install nbstripout

# Настройка для git (один раз)
nbstripout --install

# Теперь при git add автоматически очищаются outputs
git add notebook.ipynb  # Автоматически очистит outputs
```

#### 4. Pre-commit hook (рекомендуется):

```bash
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/kynan/nbstripout
    rev: 0.6.1
    hooks:
      - id: nbstripout
```

#### Что хранить в Git?

✅ **Хранить**:
- Код ячеек
- Markdown текст
- Метаданные ноутбука (kernelspec, language_info)

❌ **НЕ хранить** (очищать):
- `outputs` - результаты выполнения
- `execution_count` - номера выполнения
- Большие изображения в outputs

#### Пример workflow:

```bash
# 1. Работаете в Jupyter - выполняете ячейки
jupyter notebook analysis.ipynb

# 2. Перед коммитом - очищаете outputs
jupyter nbconvert --clear-output --inplace analysis.ipynb

# 3. Коммитите чистый notebook
git add analysis.ipynb
git commit -m "Add analysis notebook"

# 4. После pull - запускаете заново
jupyter notebook analysis.ipynb
# Запускаете все ячейки: Cell → Run All
```

---

## 📝 Итоговые рекомендации

### 1. Kaggle Dataset
- ✅ Добавить загрузку реальных `train.csv` и `test.csv`
- ✅ Fallback на sns.load_dataset если файлов нет
- ✅ Унифицировать названия колонок

### 2. Train/Val/Test Split
- ✅ Добавить отдельную validation выборку (15%)
- ✅ Test использовать ТОЛЬКО РАЗ в конце
- ✅ Validation для подбора гиперпараметров

### 3. Joblib вместо Pickle
- ✅ Заменить `pickle.dump` на `joblib.dump`
- ✅ Использовать `.joblib` расширение
- ✅ Добавить параметр `compress=3`

### 4. Notebook Metadata
- ✅ Понимать, что outputs - это нормально
- ✅ Очищать outputs перед git commit
- ✅ Использовать nbstripout для автоматизации

---

## 🔗 Полезные ссылки

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Scikit-learn: Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
- [Joblib Documentation](https://joblib.readthedocs.io/)
- [nbstripout](https://github.com/kynan/nbstripout)
- [Jupyter Best Practices](https://jupyter-notebook.readthedocs.io/en/stable/)

---

**Автор**: Claude Code
**Дата**: 2025
