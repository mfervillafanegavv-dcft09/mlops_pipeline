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