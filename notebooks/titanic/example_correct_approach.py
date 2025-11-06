"""
Titanic Analysis - Правильная реализация загрузки данных и разделения

Этот скрипт показывает правильную структуру работы с датасетом Титаник.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ (Kaggle датасет с fallback на seaborn)
# ============================================================================

def load_data():
    """Загрузка Kaggle датасета с fallback на seaborn"""

    train_path = '../../datasets/titanic/train.csv'
    test_path = '../../datasets/titanic/test.csv'

    try:
        if os.path.exists(train_path):
            # Kaggle датасет
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path) if os.path.exists(test_path) else None

            print(f"✓ Загружен Kaggle датасет")
            print(f"  Train: {df_train.shape}")
            if df_test is not None:
                print(f"  Test: {df_test.shape}")

            return df_train, df_test
        else:
            raise FileNotFoundError

    except FileNotFoundError:
        # Fallback на seaborn
        import seaborn as sns
        print("⚠ Kaggle датасет не найден, используем seaborn")
        print("  Для полноценной работы загрузите данные с Kaggle:")
        print("  https://www.kaggle.com/c/titanic/data")

        df = sns.load_dataset('titanic')

        # Переименовываем колонки для единообразия
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

        return df, None


# ============================================================================
# 2. ПРАВИЛЬНОЕ РАЗДЕЛЕНИЕ: Train / Validation / Test
# ============================================================================

def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Правильное разделение данных на Train/Validation/Test

    Train: обучаем модели
    Validation: подбираем гиперпараметры, выбираем модель
    Test: финальная оценка (используем ТОЛЬКО РАЗ!)
    """

    # Шаг 1: Отделяем test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Шаг 2: Делим оставшееся на Train и Validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted,
        random_state=random_state, stratify=y_temp
    )

    print("=" * 60)
    print("РАЗДЕЛЕНИЕ ДАННЫХ")
    print("=" * 60)
    print(f"Общий размер: {len(X)} строк")
    print(f"\n📚 Train:      {len(X_train):4d} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"🎯 Validation: {len(X_val):4d} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"🔒 Test:       {len(X_test):4d} ({len(X_test)/len(X)*100:.1f}%)")
    print("=" * 60)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# 3. СОХРАНЕНИЕ МОДЕЛИ С JOBLIB (не pickle!)
# ============================================================================

def save_model(model, scaler, model_dir='../../models'):
    """Сохранение модели с joblib"""

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'titanic_model.joblib')
    scaler_path = os.path.join(model_dir, 'titanic_scaler.joblib')

    # Сохранение с сжатием
    joblib.dump(model, model_path, compress=3)
    joblib.dump(scaler, scaler_path, compress=3)

    print(f"\n✓ Модель сохранена: {model_path}")
    print(f"✓ Scaler сохранен: {scaler_path}")

    return model_path, scaler_path


def load_model(model_dir='../../models'):
    """Загрузка модели"""

    model_path = os.path.join(model_dir, 'titanic_model.joblib')
    scaler_path = os.path.join(model_dir, 'titanic_scaler.joblib')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print(f"✓ Модель загружена: {model_path}")
    print(f"✓ Scaler загружен: {scaler_path}")

    return model, scaler


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":

    # 1. Загрузка данных
    df_train, df_test = load_data()

    # 2. Простая предобработка (для примера)
    X = df_train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].fillna(0)
    y = df_train['Survived']

    # 3. Правильное разделение
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 4. Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 5. Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 6. Оценка на validation (подбираем параметры)
    val_score = model.score(X_val_scaled, y_val)
    print(f"\n📊 Точность на Validation: {val_score:.4f}")

    # 7. Финальная оценка на test (ТОЛЬКО РАЗ!)
    test_score = model.score(X_test_scaled, y_test)
    print(f"📊 Точность на Test: {test_score:.4f}")

    # 8. Сохранение с joblib
    save_model(model, scaler)

    print("\n✅ Готово!")
