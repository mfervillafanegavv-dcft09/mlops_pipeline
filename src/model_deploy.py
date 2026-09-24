from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List

from src.ft_engineering import prepare_data
from src.model_training_evaluation import train_deployment_model


app = FastAPI(
    title="Churn Prediction API",
    description="API para predecir la probabilidad de churn de clientes",
    version="1.0.0"
)

class CustomerData(BaseModel):
    signup_month: int
    age: int
    tenure_months: int
    region: str
    channel: str
    plan: str
    sessions_week: float
    avg_session_min: float
    notif_click_rate: float
    support_tickets_3m: int
    discount_pct_3m: float
    late_payments_6m: int
    auto_renew: int

# Preparar los datos de entrenamiento
X_train, X_test, y_train, y_test, preprocessor = prepare_data(
    "Base_de_datos.csv"
)

# Entrenar el modelo que utilizaremos en producción
model = train_deployment_model(
    preprocessor,
    X_train,
    y_train
)

@app.get("/")
def home():
    return {
        "message": "Churn Prediction API funcionando correctamente"
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):

    customer_df = pd.DataFrame(
        [customer.model_dump()]
    )

    prediction = model.predict(customer_df)[0]

    probability = model.predict_proba(customer_df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }

@app.post("/predict-batch")
def predict_churn_batch(customers: List[CustomerData]):

    customers_df = pd.DataFrame(
        [customer.model_dump() for customer in customers]
    )

    predictions = model.predict(customers_df)
    probabilities = model.predict_proba(customers_df)[:, 1]

    results = []

    for prediction, probability in zip(
        predictions,
        probabilities
    ):
        results.append({
            "churn_prediction": int(prediction),
            "churn_probability": round(
                float(probability),
                4
            )
        })

    return {
        "predictions": results
    }