"""
Утилиты для загрузки и использования ML моделей
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """Класс для загрузки моделей и выполнения предсказаний"""

    def __init__(self, model_path: str = None):
        """
        Инициализация предиктора

        Args:
            model_path: Путь к сохраненной модели
        """
        self.model = None
        self.model_name = None
        self.model_path = model_path
        self.feature_names = [
            'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked',
            'family_size', 'is_alone', 'age_group', 'fare_category'
        ]

        # Маппинги для категориальных признаков
        self.sex_mapping = {'male': 1, 'female': 0}
        self.embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        Загрузка модели из файла

        Args:
            model_path: Путь к файлу модели

        Returns:
            True если модель успешно загружена
        """
        try:
            path = Path(model_path)
            if not path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            self.model = joblib.load(model_path)
            self.model_name = path.stem.replace('_model', '')
            self.model_path = model_path
            logger.info(f"Model loaded successfully: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Создание дополнительных признаков (feature engineering)

        Args:
            data: Исходные данные

        Returns:
            Данные с дополнительными признаками
        """
        df = data.copy()

        # Family size
        df['family_size'] = df['sibsp'] + df['parch'] + 1

        # Is alone
        df['is_alone'] = (df['family_size'] == 1).astype(int)

        # Age group
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 12, 18, 35, 60, 100],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

        # Fare category
        df['fare_category'] = pd.cut(
            df['fare'],
            bins=[0, 7.91, 14.45, 31, 1000],
            labels=[0, 1, 2, 3]
        ).astype(int)

        return df

    def preprocess_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Предобработка входных данных

        Args:
            input_data: Словарь с входными данными

        Returns:
            Numpy массив с предобработанными признаками
        """
        # Создаем DataFrame из входных данных
        df = pd.DataFrame([input_data])

        # Преобразуем категориальные признаки
        df['sex'] = df['sex'].map(self.sex_mapping)
        df['embarked'] = df['embarked'].map(self.embarked_mapping)

        # Создаем дополнительные признаки
        df = self._engineer_features(df)

        # Выбираем нужные признаки в правильном порядке
        features = df[self.feature_names].values

        return features

    def predict(self, input_data: Dict[str, Any]) -> Tuple[int, float]:
        """
        Выполнение предсказания

        Args:
            input_data: Словарь с входными данными

        Returns:
            Кортеж (предсказание, вероятность)
        """
        if self.model is None:
            raise ValueError("Model is not loaded")

        # Предобработка данных
        features = self.preprocess_input(input_data)

        # Предсказание
        prediction = self.model.predict(features)[0]

        # Вероятность (если модель поддерживает predict_proba)
        try:
            probabilities = self.model.predict_proba(features)[0]
            probability = probabilities[1]  # Вероятность класса 1 (выживание)
        except AttributeError:
            # Если модель не поддерживает predict_proba
            probability = float(prediction)

        return int(prediction), float(probability)

    def batch_predict(self, input_data_list: list) -> list:
        """
        Пакетное предсказание

        Args:
            input_data_list: Список словарей с входными данными

        Returns:
            Список кортежей (предсказание, вероятность)
        """
        results = []
        for input_data in input_data_list:
            prediction, probability = self.predict(input_data)
            results.append((prediction, probability))
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели

        Returns:
            Словарь с информацией о модели
        """
        if self.model is None:
            return {
                "name": "None",
                "version": "N/A",
                "features": self.feature_names,
                "model_loaded": False
            }

        return {
            "name": self.model_name,
            "version": "1.0.0",
            "features": self.feature_names,
            "model_loaded": True,
            "model_type": type(self.model).__name__
        }


def find_best_model(models_dir: str = "../models") -> str:
    """
    Поиск лучшей модели в директории

    Args:
        models_dir: Директория с моделями

    Returns:
        Путь к лучшей модели
    """
    models_path = Path(models_dir)

    # Ищем файлы моделей (joblib, pkl)
    model_files = list(models_path.glob("*.joblib")) + list(models_path.glob("*.pkl"))

    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")

    # Приоритет моделей (по названию)
    priority = ['stacking', 'xgboost', 'catboost', 'lightgbm', 'xgb', 'cat', 'lgb']

    for model_type in priority:
        for model_file in model_files:
            if model_type in model_file.stem.lower():
                logger.info(f"Found best model: {model_file.name}")
                return str(model_file)

    # Если не найдена приоритетная модель, возвращаем первую
    logger.info(f"Using first available model: {model_files[0].name}")
    return str(model_files[0])
