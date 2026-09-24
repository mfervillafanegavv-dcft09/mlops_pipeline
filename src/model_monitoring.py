import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, chi2_contingency


def simulate_current_data(reference_data, random_state=42):
    """
    Simula datos actuales con algunos cambios respecto
    de los datos históricos para probar la detección de drift.
    """
    current_data = reference_data.copy()

    rng = np.random.default_rng(random_state)

    # Simulamos cambios en variables numéricas
    current_data["sessions_week"] = (
        current_data["sessions_week"] * 0.70
    )

    current_data["support_tickets_3m"] = (
        current_data["support_tickets_3m"]
        + rng.poisson(0.5, len(current_data))
    )

    # Simulamos un cambio en una variable categórica
    current_data.loc[
        rng.random(len(current_data)) < 0.25,
        "channel"
    ] = "app"

    return current_data


def detect_numeric_drift(
    reference_data,
    current_data,
    columns,
    alpha=0.05
):
    """
    Detecta Data Drift en variables numéricas
    mediante el test Kolmogorov-Smirnov.
    """
    results = []

    for column in columns:

        statistic, p_value = ks_2samp(
            reference_data[column].dropna(),
            current_data[column].dropna()
        )

        drift_detected = p_value < alpha

        results.append({
            "Variable": column,
            "Tipo": "Numérica",
            "Métrica": "KS",
            "Estadístico": statistic,
            "p-value": p_value,
            "Drift detectado": drift_detected
        })

    return pd.DataFrame(results)


def detect_categorical_drift(
    reference_data,
    current_data,
    columns,
    alpha=0.05
):
    """
    Detecta Data Drift en variables categóricas
    mediante Chi-cuadrado.
    """
    results = []

    for column in columns:

        reference_counts = reference_data[column].value_counts()
        current_counts = current_data[column].value_counts()

        categories = reference_counts.index.union(
            current_counts.index
        )

        reference_counts = (
            reference_counts
            .reindex(categories, fill_value=0)
        )

        current_counts = (
            current_counts
            .reindex(categories, fill_value=0)
        )

        contingency_table = pd.DataFrame({
            "Referencia": reference_counts,
            "Actual": current_counts
        }).T

        statistic, p_value, _, _ = chi2_contingency(
            contingency_table
        )

        drift_detected = p_value < alpha

        results.append({
            "Variable": column,
            "Tipo": "Categórica",
            "Métrica": "Chi-cuadrado",
            "Estadístico": statistic,
            "p-value": p_value,
            "Drift detectado": drift_detected
        })

    return pd.DataFrame(results)


def run_monitoring(reference_data, current_data):
    """
    Ejecuta el monitoreo de Data Drift sobre las
    principales variables del modelo.
    """

    numeric_columns = [
        "age",
        "tenure_months",
        "sessions_week",
        "avg_session_min",
        "notif_click_rate",
        "support_tickets_3m",
        "discount_pct_3m",
        "late_payments_6m"
    ]

    categorical_columns = [
        "region",
        "channel",
        "plan"
    ]

    numeric_results = detect_numeric_drift(
        reference_data,
        current_data,
        numeric_columns
    )

    categorical_results = detect_categorical_drift(
        reference_data,
        current_data,
        categorical_columns
    )

    results = pd.concat(
        [numeric_results, categorical_results],
        ignore_index=True
    )

    return results


def streamlit_app():

    st.set_page_config(
        page_title="Monitoreo de Data Drift",
        layout="wide"
    )

    st.title("Monitoreo de Data Drift")

    st.write(
        "Comparación entre los datos históricos de referencia "
        "y los datos actuales para detectar cambios en la población."
    )

    # Cargar datos históricos
    reference_data = pd.read_csv("Base_de_datos.csv")

    # Simular datos actuales
    current_data = simulate_current_data(reference_data)

    # Ejecutar monitoreo
    drift_results = run_monitoring(
        reference_data,
        current_data
    )

    # =========================
    # RESUMEN GENERAL
    # =========================

    total_variables = len(drift_results)

    variables_with_drift = int(
        drift_results["Drift detectado"].sum()
    )

    percentage_drift = (
        variables_with_drift / total_variables
    ) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Variables monitoreadas",
        total_variables
    )

    col2.metric(
        "Variables con drift",
        variables_with_drift
    )

    col3.metric(
        "Porcentaje con drift",
        f"{percentage_drift:.1f}%"
    )

    # =========================
    # TABLA DE RESULTADOS
    # =========================

    st.subheader("Resultados del monitoreo")

    st.dataframe(
        drift_results,
        width="stretch"
    )

    # =========================
    # ALERTAS Y RECOMENDACIONES
    # =========================

    if variables_with_drift > 0:

        st.error(
            f"Alerta: se detectó Data Drift en "
            f"{variables_with_drift} variables."
        )

        affected_variables = drift_results.loc[
            drift_results["Drift detectado"],
            "Variable"
        ].tolist()

        st.write(
            "Variables afectadas:",
            ", ".join(affected_variables)
        )

        st.warning(
            "Se recomienda revisar las variables afectadas "
            "y evaluar la necesidad de reentrenar el modelo."
        )

    else:

        st.success(
            "No se detectaron cambios significativos "
            "en las variables monitoreadas."
        )

    # =========================
    # COMPARACIÓN DE DISTRIBUCIONES
    # =========================

    st.subheader("Comparación de distribuciones")

    numeric_columns = [
        "age",
        "tenure_months",
        "sessions_week",
        "avg_session_min",
        "notif_click_rate",
        "support_tickets_3m",
        "discount_pct_3m",
        "late_payments_6m"
    ]

    selected_variable = st.selectbox(
        "Seleccionar variable numérica",
        numeric_columns
    )

    fig, ax = plt.subplots()

    ax.hist(
        reference_data[selected_variable],
        bins=20,
        alpha=0.5,
        label="Histórico"
    )

    ax.hist(
        current_data[selected_variable],
        bins=20,
        alpha=0.5,
        label="Actual"
    )

    ax.set_title(
        f"Distribución histórica vs. actual: "
        f"{selected_variable}"
    )

    ax.set_xlabel(selected_variable)
    ax.set_ylabel("Frecuencia")
    ax.legend()

    st.pyplot(fig)

    # =========================
    # COMPARACIÓN CATEGÓRICA
    # =========================

    st.subheader("Comparación de variables categóricas")

    categorical_columns = [
        "region",
        "channel",
        "plan"
    ]

    selected_category = st.selectbox(
        "Seleccionar variable categórica",
        categorical_columns
    )

    reference_pct = (
        reference_data[selected_category]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    current_pct = (
        current_data[selected_category]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    category_comparison = pd.DataFrame({
        "Histórico (%)": reference_pct,
        "Actual (%)": current_pct
    }).fillna(0)

    st.bar_chart(category_comparison)

    # =========================
    # SEMÁFORO GENERAL
    # =========================

    st.subheader("Estado general del monitoreo")

    if percentage_drift == 0:
        st.success(
            "🟢 Estado estable: no se detectó Data Drift."
        )

    elif percentage_drift <= 20:
        st.warning(
            "🟡 Estado de atención: se detectó drift "
            "en una proporción reducida de variables."
        )

    else:
        st.error(
            "🔴 Estado crítico: se detectó drift en más del "
            "20% de las variables monitoreadas. "
            "Se recomienda revisar el modelo y evaluar su reentrenamiento."
        )
        

if __name__ == "__main__":
    streamlit_app()