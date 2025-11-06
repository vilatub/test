# Анализ датасета Титаник

Подробный анализ датасета пассажиров Титаника с применением машинного обучения.

## Описание

Проект предсказывает выживаемость пассажиров Титаника на основе их характеристик (пол, возраст, класс билета, и т.д.).

## Файлы

- `titanic_analysis.ipynb` - основной ноутбук с полным анализом
- `example_correct_approach.py` - пример правильной структуры кода

## Данные

Датасет можно получить с [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data):
- `train.csv` - обучающие данные (891 строка)
- `test.csv` - тестовые данные для submission (418 строк)

Разместите файлы в `../../datasets/titanic/`

## Использование

### Запуск ноутбука

```bash
jupyter notebook titanic_analysis.ipynb
```

### Запуск примера

```bash
python example_correct_approach.py
```

## Структура анализа

1. Исследовательский анализ данных (EDA)
2. Визуализация данных
3. Предобработка и Feature Engineering
4. Обучение моделей машинного обучения
5. Сравнение и оптимизация моделей

## Модели

Сравниваются 6 моделей:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM
- KNN

Лучший результат: Random Forest (~82-85% accuracy)

## Технологии

- pandas, numpy - работа с данными
- matplotlib, seaborn - визуализация
- scikit-learn - машинное обучение
- joblib - сохранение моделей

## Требования

```bash
pip install -r ../../requirements.txt
```
