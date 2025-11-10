"""
FastAPI приложение для предсказания выживаемости на Титанике
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from pathlib import Path

from schemas import (
    PassengerInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse
)
from model_utils import ModelPredictor, find_best_model

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API для предсказания выживаемости пассажиров Титаника с использованием ML моделей",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальная переменная для предиктора
predictor: ModelPredictor = None


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    global predictor
    try:
        # Ищем лучшую модель
        models_dir = Path(__file__).parent.parent / "models"
        logger.info(f"Looking for models in: {models_dir}")

        model_path = find_best_model(str(models_dir))
        predictor = ModelPredictor(model_path)

        logger.info("Application started successfully")
        logger.info(f"Model loaded: {predictor.model_name}")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.warning("Application started without model")
        predictor = ModelPredictor()


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении работы"""
    logger.info("Application shutting down")


@app.get("/", tags=["Root"])
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Titanic Survival Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check эндпоинт для проверки состояния сервиса
    """
    model_loaded = predictor is not None and predictor.model is not None

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version="1.0.0",
        model_loaded=model_loaded
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Получение информации о загруженной модели
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )

    info = predictor.get_model_info()

    return ModelInfo(
        name=info.get("name", "Unknown"),
        version=info.get("version", "1.0.0"),
        features=info.get("features", []),
        training_date=None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_survival(passenger: PassengerInput):
    """
    Предсказание выживаемости для одного пассажира

    Args:
        passenger: Данные пассажира

    Returns:
        Предсказание и вероятность выживания
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )

    try:
        # Преобразуем Pydantic модель в словарь
        input_data = {
            "pclass": passenger.pclass,
            "sex": passenger.sex.value,
            "age": passenger.age,
            "sibsp": passenger.sibsp,
            "parch": passenger.parch,
            "fare": passenger.fare,
            "embarked": passenger.embarked.value
        }

        # Получаем предсказание
        survived, probability = predictor.predict(input_data)

        return PredictionResponse(
            survived=survived,
            probability=round(probability, 4),
            model_used=predictor.model_name
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Пакетное предсказание выживаемости для нескольких пассажиров

    Args:
        request: Список данных пассажиров

    Returns:
        Список предсказаний
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )

    try:
        predictions = []

        for passenger in request.passengers:
            input_data = {
                "pclass": passenger.pclass,
                "sex": passenger.sex.value,
                "age": passenger.age,
                "sibsp": passenger.sibsp,
                "parch": passenger.parch,
                "fare": passenger.fare,
                "embarked": passenger.embarked.value
            }

            survived, probability = predictor.predict(input_data)

            predictions.append(
                PredictionResponse(
                    survived=survived,
                    probability=round(probability, 4),
                    model_used=predictor.model_name
                )
            )

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Обработчик HTTP исключений"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Обработчик общих исключений"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="Internal server error",
            error_code="INTERNAL_ERROR"
        ).dict()
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
