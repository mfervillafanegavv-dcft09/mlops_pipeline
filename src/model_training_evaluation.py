import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from ft_engineering import prepare_data


def build_model(preprocessor, classifier):
    """
    Construye un pipeline que combina el preprocesamiento
    de los datos con un modelo de clasificación.
    """
    model = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", classifier)
    ])

    return model


def summarize_classification(model_name, model, X_test, y_test):
    """
    Evalúa un modelo de clasificación y devuelve
    sus principales métricas.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Modelo": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    return metrics


# Carga y preparación de los datos
X_train, X_test, y_train, y_test, preprocessor = prepare_data(
    "Base_de_datos.csv"
)


# Definición de los modelos
classifiers = {
    "Regresión Logística": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# Entrenamiento y evaluación
results = []

for model_name, classifier in classifiers.items():

    model = build_model(preprocessor, classifier)

    model.fit(X_train, y_train)

    metrics = summarize_classification(
        model_name,
        model,
        X_test,
        y_test
    )

    results.append(metrics)


# Tabla comparativa
results_df = pd.DataFrame(results)

print("\n--- Comparación de modelos ---")
print(results_df.round(4))


# Gráfico comparativo
results_plot = results_df.set_index("Modelo")

results_plot[
    ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Comparación de métricas por modelo")
plt.ylabel("Valor de la métrica")
plt.xlabel("Modelo")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Métrica")
plt.tight_layout()
plt.show()