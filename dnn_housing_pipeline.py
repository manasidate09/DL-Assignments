import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras    
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# 1. Load dataset
data = pd.read_csv('/mnt/data/AmesHousing.csv')

# 2. Drop rows with missing target and select a subset of features
data = data.dropna(subset=["SalePrice"])
numeric_features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
data = data[numeric_features + ["SalePrice"]].dropna()

# 3. Split into train/test
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# 4. Separate features and labels
train_labels = train_df.pop("SalePrice")
test_labels = test_df.pop("SalePrice")

# 5. Normalize numeric features
normalizer = layers.Normalization()
normalizer.adapt(np.array(train_df))

# 6. Build a functional model
inputs = keras.Input(shape=(len(numeric_features),))
x = normalizer(inputs)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(1)(x)

model = keras.Model(inputs, outputs)

# 7. Compile and train
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(
    train_df, train_labels,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1
)

# 8. Evaluate
loss, mae = model.evaluate(test_df, test_labels)
print(f"Test MAE: {mae:.2f}")
