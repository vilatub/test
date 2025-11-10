"""
Pydantic схемы для валидации запросов и ответов API
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum


class Sex(str, Enum):
    """Пол пассажира"""
    male = "male"
    female = "female"


class Embarked(str, Enum):
    """Порт посадки"""
    C = "C"  # Cherbourg
    Q = "Q"  # Queenstown
    S = "S"  # Southampton


class PassengerInput(BaseModel):
    """Схема входных данных для предсказания выживаемости на Титанике"""
    pclass: int = Field(..., ge=1, le=3, description="Класс билета (1, 2, 3)")
    sex: Sex = Field(..., description="Пол пассажира")
    age: float = Field(..., ge=0, le=100, description="Возраст пассажира")
    sibsp: int = Field(..., ge=0, description="Количество братьев/сестер/супругов на борту")
    parch: int = Field(..., ge=0, description="Количество родителей/детей на борту")
    fare: float = Field(..., ge=0, description="Стоимость билета")
    embarked: Embarked = Field(..., description="Порт посадки")

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Возраст должен быть в диапазоне 0-100')
        return v

    @field_validator('fare')
    @classmethod
    def validate_fare(cls, v):
        if v < 0:
            raise ValueError('Стоимость билета не может быть отрицательной')
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pclass": 3,
                    "sex": "male",
                    "age": 22.0,
                    "sibsp": 1,
                    "parch": 0,
                    "fare": 7.25,
                    "embarked": "S"
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Схема ответа с предсказанием"""
    survived: int = Field(..., description="Предсказание выживания (0 - нет, 1 - да)")
    probability: float = Field(..., ge=0, le=1, description="Вероятность выживания")
    model_used: str = Field(..., description="Использованная модель")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "survived": 0,
                    "probability": 0.23,
                    "model_used": "XGBoost"
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """Схема для пакетного предсказания"""
    passengers: List[PassengerInput] = Field(..., min_length=1, max_length=100)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "passengers": [
                        {
                            "pclass": 3,
                            "sex": "male",
                            "age": 22.0,
                            "sibsp": 1,
                            "parch": 0,
                            "fare": 7.25,
                            "embarked": "S"
                        },
                        {
                            "pclass": 1,
                            "sex": "female",
                            "age": 38.0,
                            "sibsp": 1,
                            "parch": 0,
                            "fare": 71.28,
                            "embarked": "C"
                        }
                    ]
                }
            ]
        }
    }


class BatchPredictionResponse(BaseModel):
    """Схема ответа для пакетного предсказания"""
    predictions: List[PredictionResponse]
    total_count: int = Field(..., description="Общее количество предсказаний")


class ModelInfo(BaseModel):
    """Информация о модели"""
    name: str = Field(..., description="Название модели")
    version: str = Field(..., description="Версия модели")
    accuracy: Optional[float] = Field(None, description="Точность модели на валидационной выборке")
    features: List[str] = Field(..., description="Список признаков, используемых моделью")
    training_date: Optional[str] = Field(None, description="Дата обучения модели")


class HealthResponse(BaseModel):
    """Схема ответа health check"""
    status: str = Field(..., description="Статус сервиса")
    version: str = Field(..., description="Версия API")
    model_loaded: bool = Field(..., description="Статус загрузки модели")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "1.0.0",
                    "model_loaded": True
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Схема ответа с ошибкой"""
    detail: str = Field(..., description="Описание ошибки")
    error_code: Optional[str] = Field(None, description="Код ошибки")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detail": "Model not loaded",
                    "error_code": "MODEL_NOT_LOADED"
                }
            ]
        }
    }
