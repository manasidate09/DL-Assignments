import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 1. Load and preprocess data
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    
    # Drop missing target values
    data = data.dropna(subset=["SalePrice"])
    
    # Select numeric features
    numeric_features = ["Overall Qual", "Gr Liv Area", "Garage Cars", 
                        "Total Bsmt SF", "Full Bath", "Year Built"]
    
    # drop SalePrice 
    data = data[numeric_features + ["SalePrice"]].dropna()
    
    # Split train/test
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    
    # Separate features and labels
    train_labels = train_df.pop("SalePrice")
    test_labels = test_df.pop("SalePrice")
    
    return train_df, test_df, train_labels, test_labels, numeric_features


# 2. Build model
def build_model(numeric_features, train_df):
    normalizer = layers.Normalization()
    normalizer.adapt(np.array(train_df))
    
    inputs = keras.Input(shape=(len(numeric_features),))
    x = normalizer(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs)
    return model
def save_model(model, path="saved_model"):
    model.save(path)
    
def load_model(path="saved_model"):
    return keras.models.load_model(path)

def predict(model, input_data, numeric_features):
    # Ensure the order of features matches training
    input_df = pd.DataFrame([input_data])[numeric_features]
    return model.predict(input_df)[0][0]

# 3. Compile and train model
def compile_and_train_model(model, train_df, train_labels):
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    history = model.fit(
        train_df, train_labels,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history


# 4. Evaluate model
def evaluate_model(model, test_df, test_labels):
    loss, mae = model.evaluate(test_df, test_labels)
    print(f"Test MAE: {mae:.2f}")
    print(f"Loss: {loss:.2f}")



# === Main Pipeline ===
def custom_pipeline(filepath):
    train_df, test_df, train_labels, test_labels, numeric_features = load_and_preprocess_data(filepath)
    model = build_model(numeric_features, train_df)
    model.summary()
    history = compile_and_train_model(model, train_df, train_labels)
    evaluate_model(model, test_df, test_labels)
    save_model(model) 
    return model


# === Run ===
model = custom_pipeline("AmesHousing.csv")
