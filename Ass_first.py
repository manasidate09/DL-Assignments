import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# Setting random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Function to load and clean data
def load_and_clean_data(file_path='AmesHousing.csv'):
    df = pd.read_csv(file_path)
    
    # Dropping irrelevant columns
    columns_to_drop = ['Order', 'PID', 'Alley', 'Pool QC', 'Fence', 'Misc Feature']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Converting string booleans to numeric
    if 'Central Air' in df.columns:
        df['Central Air'] = df['Central Air'].map({'Y': 1, 'N': 0})
    
    # Handling missing values
    if 'Lot Frontage' in df.columns:
        df['Lot Frontage'] = df['Lot Frontage'].fillna(df['Lot Frontage'].median())
    
    if 'Garage Yr Blt' in df.columns:
        df['Garage Yr Blt'] = pd.to_numeric(df['Garage Yr Blt'], errors='coerce')
        df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(df['Year Built'])
    
    return df

# Function to identify feature columns
def get_feature_columns(df):
    numeric_features = [
        'Lot Frontage', 'Lot Area', 'Year Built', 'Year Remod/Add', 'Mas Vnr Area',
        'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF',
        '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath',
        'Full Bath', 'Half Bath', 'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd',
        'Fireplaces', 'Garage Yr Blt', 'Garage Cars', 'Garage Area', 'Wood Deck SF',
        'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val'
    ]
    
    categorical_features = [
        'MS Zoning', 'Street', 'Lot Shape', 'Land Contour', 'Utilities', 'Lot Config',
        'Land Slope', 'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type',
        'House Style', 'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd',
        'Mas Vnr Type', 'Exter Qual', 'Exter Cond', 'Foundation', 'Bsmt Qual',
        'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2', 'Heating',
        'Heating QC', 'Electrical', 'Kitchen Qual', 'Functional', 'Fireplace Qu',
        'Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond', 'Paved Drive',
        'Sale Type', 'Sale Condition'
    ]
    
    # Filtering columns present in the dataset
    numeric_features = [col for col in numeric_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]
    
    return numeric_features, categorical_features

# Function to preprocess numeric features
def preprocess_numeric_features(X, numeric_features, imputer=None, scaler=None, fit=True):
    X_numeric = X[numeric_features].copy()
    
    if fit:
        imputer = SimpleImputer(strategy='median')
        X_numeric = imputer.fit_transform(X_numeric)
        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(X_numeric)
    else:
        if imputer is None or scaler is None:
            raise ValueError("Imputer and scaler must be provided when fit=False")
        X_numeric = imputer.transform(X_numeric)
        X_numeric = scaler.transform(X_numeric)
    
    return X_numeric, imputer, scaler

# Function to preprocess categorical features
def preprocess_categorical_features(X, categorical_features, imputer=None, encoder=None, fit=True):
    X_categorical = X[categorical_features].copy()
    
    if fit:
        imputer = SimpleImputer(strategy='constant', fill_value='missing')
        X_categorical = imputer.fit_transform(X_categorical)
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_categorical = encoder.fit_transform(X_categorical)
    else:
        if imputer is None or encoder is None:
            raise ValueError("Imputer and encoder must be provided when fit=False")
        X_categorical = imputer.transform(X_categorical)
        X_categorical = encoder.transform(X_categorical)
    
    return X_categorical, imputer, encoder

# Function to combine features
def combine_features(X_numeric, X_categorical):
    return np.hstack((X_numeric, X_categorical))

# Function to build DNN model
def build_dnn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Main functional pipeline
def housing_price_prediction_pipeline(file_path='AmesHousing.csv'):
    # Step 1: Load and clean data
    df = load_and_clean_data(file_path)
    
    # Step 2: Separate features and target
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 4: Get feature columns
    numeric_features, categorical_features = get_feature_columns(X)
    
    # Step 5: Preprocess numeric features
    X_train_numeric, num_imputer, scaler = preprocess_numeric_features(X_train, numeric_features, fit=True)
    X_test_numeric, _, _ = preprocess_numeric_features(X_test, numeric_features, imputer=num_imputer, scaler=scaler, fit=False)
    
    # Step 6: Preprocess categorical features
    X_train_categorical, cat_imputer, encoder = preprocess_categorical_features(X_train, categorical_features, fit=True)
    X_test_categorical, _, _ = preprocess_categorical_features(X_test, categorical_features, imputer=cat_imputer, encoder=encoder, fit=False)
    
    # Step 7: Combine features
    X_train_processed = combine_features(X_train_numeric, X_train_categorical)
    X_test_processed = combine_features(X_test_numeric, X_test_categorical)
    
    # Step 8: Build and train model
    model = build_dnn_model(input_dim=X_train_processed.shape[1])
    
    history = model.fit(
        X_train_processed, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=1
    )
    
    # Step 9: Evaluate model
    test_loss, test_mae = model.evaluate(X_test_processed, y_test, verbose=0)
    print(f"\nTest MAE: ${test_mae:.2f}")
    
    # Step 10: Save model and preprocessors
    model.save('housing_price_model.h5')
    joblib.dump({'num_imputer': num_imputer, 'scaler': scaler, 'cat_imputer': cat_imputer, 'encoder': encoder}, 'preprocessors.pkl')
    
    return model, history

# Running the pipeline
if __name__ == "__main__":
    model, history = housing_price_prediction_pipeline()