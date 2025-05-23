import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers, models

# === Pipeline Functions ===

def load_data(path):
    df = pd.read_csv(path)
    df = shuffle(df, random_state=42)
    return df

def preprocess_data(df):
    df = df.dropna(subset=["SalePrice"])
    X = df.drop(columns=["SalePrice", "Order", "PID"], errors='ignore')
    y = df["SalePrice"].values.reshape(-1, 1)

    numeric_cols = X.select_dtypes(include=["number"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    return X_processed, y_scaled, preprocessor, y_scaler

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def build_model(input_shape):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Linear output for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# === Main Execution ===

if __name__ == "__main__":
    # Load and process data
    path = "AmesHousing.csv"  # Update this if the file path is different
    df = load_data(path)
    X, y, preprocessor, y_scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model(X_train.shape[1])
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X, y),
        epochs=10,
        batch_size=32
    )
    # Build and train the model
    '''model = build_model(X_train.shape[1])
    history = model.fit(
        X_train, y_train,
        validation_data=(X, y),
        epochs=50,
        batch_size=32,
        verbose=1
    )'''

    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae:.2f}")
