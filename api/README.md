# Titanic ML API

REST API для предсказания выживаемости пассажиров Титаника с использованием машинного обучения.

## 🚀 Быстрый старт

### Локальный запуск

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Убедитесь, что модель находится в папке `models/`:
```bash
ls ../models/
```

3. Запустите сервер:
```bash
cd api
python main.py
```

Или с помощью uvicorn:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

4. Откройте документацию: http://localhost:8000/docs

### Запуск с Docker

1. Соберите образ:
```bash
docker build -t titanic-ml-api .
```

2. Запустите контейнер:
```bash
docker run -p 8000:8000 titanic-ml-api
```

### Запуск с Docker Compose

```bash
docker-compose up -d
```

Проверка статуса:
```bash
docker-compose ps
```

Просмотр логов:
```bash
docker-compose logs -f api
```

Остановка:
```bash
docker-compose down
```

## 📡 API Endpoints

### Health Check
```http
GET /health
```

Ответ:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true
}
```

### Информация о модели
```http
GET /model/info
```

Ответ:
```json
{
  "name": "xgboost",
  "version": "1.0.0",
  "features": ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "family_size", "is_alone", "age_group", "fare_category"],
  "training_date": null
}
```

### Предсказание (одиночное)
```http
POST /predict
Content-Type: application/json

{
  "pclass": 3,
  "sex": "male",
  "age": 22.0,
  "sibsp": 1,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S"
}
```

Ответ:
```json
{
  "survived": 0,
  "probability": 0.2341,
  "model_used": "xgboost"
}
```

### Предсказание (пакетное)
```http
POST /predict/batch
Content-Type: application/json

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
```

Ответ:
```json
{
  "predictions": [
    {
      "survived": 0,
      "probability": 0.2341,
      "model_used": "xgboost"
    },
    {
      "survived": 1,
      "probability": 0.9124,
      "model_used": "xgboost"
    }
  ],
  "total_count": 2
}
```

## 🔍 Примеры использования

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Предсказание
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 29.0,
    "sibsp": 0,
    "parch": 0,
    "fare": 211.33,
    "embarked": "S"
  }'
```

### Python (requests)

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "pclass": 1,
    "sex": "female",
    "age": 29.0,
    "sibsp": 0,
    "parch": 0,
    "fare": 211.33,
    "embarked": "S"
}

response = requests.post(url, json=data)
print(response.json())
```

### JavaScript (fetch)

```javascript
const url = 'http://localhost:8000/predict';
const data = {
  pclass: 1,
  sex: 'female',
  age: 29.0,
  sibsp: 0,
  parch: 0,
  fare: 211.33,
  embarked: 'S'
};

fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(data)
})
  .then(response => response.json())
  .then(data => console.log(data));
```

## 📋 Схема данных

### PassengerInput

| Поле | Тип | Описание | Ограничения |
|------|-----|----------|-------------|
| pclass | int | Класс билета | 1, 2, 3 |
| sex | str | Пол пассажира | "male", "female" |
| age | float | Возраст | 0-100 |
| sibsp | int | Братья/сестры/супруги на борту | >= 0 |
| parch | int | Родители/дети на борту | >= 0 |
| fare | float | Стоимость билета | >= 0 |
| embarked | str | Порт посадки | "C", "Q", "S" |

### PredictionResponse

| Поле | Тип | Описание |
|------|-----|----------|
| survived | int | Предсказание (0 или 1) |
| probability | float | Вероятность выживания (0-1) |
| model_used | str | Название использованной модели |

## 🛠 Технологии

- **FastAPI** - современный веб-фреймворк для создания API
- **Pydantic** - валидация данных
- **scikit-learn** - ML модели
- **XGBoost/LightGBM/CatBoost** - градиентный бустинг
- **uvicorn** - ASGI сервер
- **Docker** - контейнеризация

## 📊 Feature Engineering

API автоматически создает следующие признаки:

1. **family_size** = sibsp + parch + 1
2. **is_alone** = 1 если family_size == 1, иначе 0
3. **age_group** = категории возраста [0-12, 12-18, 18-35, 35-60, 60-100]
4. **fare_category** = категории стоимости билета

## 🐛 Troubleshooting

### Модель не загружена

Убедитесь, что в папке `models/` есть обученная модель (.pkl или .joblib файл):
```bash
ls models/
```

### Порт уже занят

Измените порт в команде запуска:
```bash
uvicorn api.main:app --port 8001
```

### Проблемы с Docker

Проверьте логи:
```bash
docker logs titanic-ml-api
```

Пересоберите образ:
```bash
docker-compose build --no-cache
docker-compose up -d
```

## 📝 Лицензия

MIT
