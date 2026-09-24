import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def simulate_current_data(reference_data, random_state=42):
    """
    Simula datos actuales para demostrar un escenario
    de Data Drift.
    """
    current_data = reference_data.copy()

    rng = np.random.default_rng(random_state)

    current_data["sessions_week"] = (
        current_data["sessions_week"] * 0.70
    )

    current_data["support_tickets_3m"] = (
        current_data["support_tickets_3m"]
        + rng.poisson(0.5, len(current_data))
    )

    return current_data

def detect_drift(reference_data, current_data, columns, threshold=20):
    """
    Compara la media de variables numéricas entre los datos
    de referencia y los datos actuales.

    Se considera una alerta de drift cuando el cambio
    porcentual supera el umbral establecido.
    """
    results = []

    for column in columns:
        reference_mean = reference_data[column].mean()
        current_mean = current_data[column].mean()

        change_pct = abs(
            (current_mean - reference_mean) / reference_mean
        ) * 100

        drift_detected = change_pct > threshold

        results.append({
            "Variable": column,
            "Media referencia": reference_mean,
            "Media actual": current_mean,
            "Cambio (%)": change_pct,
            "Drift detectado": drift_detected
        })

    return pd.DataFrame(results)

if __name__ == "__main__":

    reference_data = pd.read_csv("Base_de_datos.csv")

    current_data = simulate_current_data(reference_data)

    columns_to_monitor = [
        "sessions_week",
        "support_tickets_3m"
    ]

    drift_results = detect_drift(
        reference_data,
        current_data,
        columns_to_monitor
    )

    print("\n--- Monitoreo de Data Drift ---")
    print(drift_results.round(3))