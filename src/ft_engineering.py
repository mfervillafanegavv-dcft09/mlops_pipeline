import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def prepare_data(data_path):
    df = pd.read_csv(data_path)

    X = df.drop(columns=["churn", "customer_id"])
    y = df["churn"]

    numeric_features = [
        "signup_month",
        "age",
        "tenure_months",
        "sessions_week",
        "avg_session_min",
        "notif_click_rate",
        "support_tickets_3m",
        "discount_pct_3m",
        "late_payments_6m",
        "auto_renew"
    ]

    nominal_features = [
        "region",
        "channel"
    ]

    ordinal_features = [
        "plan"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    nominal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=[["Basic", "Plus", "Premium"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("nom", nominal_transformer, nominal_features),
            ("ord", ordinal_transformer, ordinal_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        "Base_de_datos.csv"
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)