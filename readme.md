# MLOps Pipeline - Predicción de Churn

## Descripción del proyecto

Este proyecto desarrolla un pipeline de Machine Learning y MLOps orientado a la predicción de churn de clientes.

El objetivo es identificar clientes con riesgo de abandono a partir de información histórica relacionada con su comportamiento, antigüedad, plan contratado, actividad, interacción con notificaciones, tickets de soporte y pagos.

El proyecto abarca el proceso completo desde la exploración de los datos hasta el despliegue y monitoreo del modelo.

---

## Caso de negocio

El churn representa la pérdida de clientes de una empresa.

Detectar anticipadamente clientes con mayor probabilidad de abandono permite implementar estrategias de retención, priorizar acciones comerciales y reducir la pérdida de ingresos.

La variable objetivo utilizada es:

- `churn = 0`: el cliente permanece.
- `churn = 1`: el cliente abandona.

El dataset contiene 5.000 registros y presenta un desbalance moderado:

- 75,64 % de clientes sin churn.
- 24,36 % de clientes con churn.

---

## Estructura del proyecto

```text
mlops_pipeline/
├── src/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_deploy.py
│   └── model_monitoring.py
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
├── .dockerignore
├── Dockerfile
└── readme.md
```

---

## 1. Carga y comprensión de los datos

En `Cargar_datos.ipynb` se realiza:

- Carga del dataset.
- Revisión de dimensiones.
- Análisis de tipos de datos.
- Detección de valores nulos.
- Detección de registros duplicados.
- Revisión inicial de las variables.

El dataset contiene 5.000 observaciones y 15 variables, sin valores nulos ni registros duplicados.

---

## 2. Análisis Exploratorio de Datos (EDA)

En `comprension_eda.ipynb` se desarrollan análisis:

- Univariados.
- Bivariados.
- Multivariados.
- Distribución de la variable objetivo.
- Correlaciones.
- Comparaciones entre variables y churn.

Entre los principales hallazgos se observaron diferencias asociadas con variables como:

- Antigüedad del cliente.
- Edad.
- Cantidad de sesiones semanales.
- Tickets de soporte.
- Renovación automática.
- Tipo de plan.

---

## 3. Feature Engineering

El archivo `ft_engineering.py` contiene el procesamiento de las variables previo al entrenamiento.

Se utiliza `ColumnTransformer` para aplicar diferentes transformaciones según el tipo de variable.

### Variables numéricas

Se utiliza `SimpleImputer` con estrategia de mediana.

### Variables categóricas nominales

Las variables `region` y `channel` son procesadas mediante:

- `SimpleImputer`.
- `OneHotEncoder`.

### Variable categórica ordinal

Para `plan` se utiliza `OrdinalEncoder` respetando el orden:

```text
Basic < Plus < Premium
```

El identificador `customer_id` se elimina del conjunto de variables predictoras.

Los datos se dividen en entrenamiento y prueba mediante `train_test_split`, utilizando estratificación de la variable objetivo.

---

## 4. Entrenamiento y evaluación

Se evaluaron tres algoritmos de clasificación supervisada:

- Regresión Logística.
- Random Forest.
- Gradient Boosting.

Las métricas utilizadas fueron:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.

Resultados obtenidos:

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Regresión Logística | 0.752 | 0.4737 | 0.1475 | 0.2250 | 0.7288 |
| Random Forest | 0.733 | 0.4455 | 0.3852 | 0.4132 | 0.6910 |
| Gradient Boosting | 0.759 | 0.5195 | 0.1639 | 0.2492 | 0.7086 |

Para el despliegue se utiliza **Random Forest**, priorizando su mayor Recall y F1-score entre los modelos evaluados, debido a la importancia de identificar clientes que efectivamente presentan riesgo de churn.

---

## 5. Monitoreo y Data Drift

El archivo `model_monitoring.py` implementa mecanismos de detección de Data Drift.

Se utilizan:

- Test de Kolmogorov-Smirnov para variables numéricas.
- Test Chi-cuadrado para variables categóricas.

El monitoreo compara datos históricos de referencia con datos actuales simulados.

Para demostrar el funcionamiento del sistema se introducen cambios controlados en determinadas variables. El monitor detecta correctamente drift en:

- `sessions_week`.
- `support_tickets_3m`.
- `channel`.

La simulación se utiliza únicamente con fines demostrativos para validar el mecanismo de monitoreo.

---

## 6. Aplicación Streamlit

`model_monitoring.py` incluye una aplicación desarrollada con Streamlit para visualizar el estado del monitoreo.

El dashboard permite observar:

- Cantidad de variables monitoreadas.
- Cantidad y porcentaje de variables con drift.
- Tabla de métricas.
- Alertas automáticas.
- Comparación de distribuciones históricas y actuales.
- Comparación de variables categóricas.
- Indicador general de riesgo.
- Recomendaciones ante la detección de drift.

Para ejecutar:

```bash
streamlit run src/model_monitoring.py
```

---

## 7. API de predicción

El archivo `model_deploy.py` implementa una API mediante FastAPI.

La API permite realizar predicciones de churn y obtener la probabilidad estimada.

Para ejecutar localmente:

```bash
uvicorn src.model_deploy:app --reload
```

La documentación interactiva de la API puede consultarse en:

```text
http://127.0.0.1:8000/docs
```

Se implementan endpoints para predicción individual y predicción por lotes.

---

## 8. Docker

El proyecto incluye:

- `Dockerfile`.
- `.dockerignore`.
- `requirements.txt`.

Estos archivos permiten contenerizar la aplicación junto con sus dependencias y el servidor Uvicorn.

Para construir la imagen:

```bash
docker build -t mlops-churn .
```

Para ejecutar el contenedor:

```bash
docker run -p 8000:8000 mlops-churn
```

---

## 9. Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- FastAPI
- Uvicorn
- Streamlit
- Docker
- Git
- GitHub

---

## Versionado

El proyecto utiliza las siguientes ramas:

- `developer`: desarrollo.
- `certification`: validación de avances.
- `master`: versión principal.

El flujo de trabajo permite separar las etapas de desarrollo, validación y publicación.

---

## Conclusión

El proyecto integra las distintas etapas del ciclo de vida de un modelo de Machine Learning: análisis exploratorio, preparación de datos, entrenamiento, evaluación, despliegue y monitoreo.

Además de generar predicciones de churn, se incorporan prácticas de MLOps orientadas a reproducibilidad, versionado, despliegue mediante API y detección de cambios en las distribuciones de los datos.