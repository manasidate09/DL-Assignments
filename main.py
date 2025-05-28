import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv("AmesHousing.csv")

# Drop irrelevant or high-missing columns
df = df.drop(columns=["Order", "PID"], errors="ignore")

# Split features and target
X = df.drop("SalePrice", axis=1)
y = np.log1p(df["SalePrice"])
#y = df["SalePrice"]

# Identify column types
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# ---------- Custom Functional Pipeline ----------

def impute_numerical(X, strategy="median"):
    imputed = X.copy()
    for col in numerical_cols:
        if strategy == "median":
            fill_val = imputed[col].median()
        elif strategy == "mean":
            fill_val = imputed[col].mean()
        #imputed[col].fillna(fill_val, inplace=True)
        imputed[col] = imputed[col].fillna(fill_val)
    return imputed

def impute_categorical(X):
    imputed = X.copy()
    for col in categorical_cols:
        fill_val = imputed[col].mode()[0]
        #imputed[col].fillna(fill_val, inplace=True)
        imputed[col] = imputed[col].fillna(fill_val)
    return imputed

def encode_categorical(X):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    #encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(X[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)
    return encoded_df, encoder

def scale_numerical(X):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X[numerical_cols])
    scaled_df = pd.DataFrame(scaled, columns=numerical_cols, index=X.index)
    return scaled_df, scaler

# Preprocess
X_num_imputed = impute_numerical(X)
X_cat_imputed = impute_categorical(X)

X_num_scaled, num_scaler = scale_numerical(X_num_imputed)
X_cat_encoded, cat_encoder = encode_categorical(X_cat_imputed)

# Combine
X_processed = pd.concat([X_num_scaled, X_cat_encoded], axis=1)
print(X_processed.columns)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# ---------- DNN Model ----------

model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mae", metrics=["mae"])

early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
loss, mae = model.evaluate(X_val, y_val)
print(f"Validation MAE (log scale): {mae:.4f}")

# Predict and invert log transform
y_pred_log = model.predict(X_val)
y_pred = np.expm1(y_pred_log).flatten()
y_actual = np.expm1(y_val).values

# Final MAE on original scale
final_mae = np.mean(np.abs(y_pred - y_actual))
print(f"Validation MAE (original): {final_mae:.2f}")
