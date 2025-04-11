import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib


# Flag to choose feature set
use_all_features = False  # Set to True for all features, False for "Change Adj Close" and "Volume"

# Load your combined data
combined_data = pd.read_csv("/app/data/new_normalized_combined_data.csv")

# Fit and save scaler for all features
scaler_all_features = MinMaxScaler()
columns_all_features = ['Adj Close', 'Open', 'High', 'Low', 'Volume']
scaler_all_features.fit(combined_data[columns_all_features])
joblib.dump(scaler_all_features, "/app/models/scaler_all_features.pkl")

# Fit and save scaler for "Change Adj Close" and "Volume"
scaler_change_volume = MinMaxScaler()
columns_change_volume = ['Change Adj Close', 'Volume']
combined_data['Change Adj Close'] = combined_data['Adj Close'].diff().fillna(0)
scaler_change_volume.fit(combined_data[columns_change_volume])
joblib.dump(scaler_change_volume, "/app/models/scaler_change_volume.pkl")



# File paths for scalers
SCALER_ALL_FEATURES_FILE = "/app/models/scaler_all_features.pkl"
SCALER_CHANGE_VOLUME_FILE = "/app/models/scaler_change_volume.pkl"
MODEL_FILE = "/app/models/20250410_model_v5.h5"


# Load the appropriate scaler
if use_all_features:
    scaler = joblib.load(SCALER_ALL_FEATURES_FILE)
    # ['Adj Close', 'Open', 'High', 'Low', 'Volume']
    new_values = np.array([
        [214.13, 212.35, 214.76, 212.35, 3618437],  # Time step 1 (oldest)
        [214.06, 214.11, 214.29, 213.67, 12186227],  # Time step 2
        [210.8, 212.18, 214.07, 210.76, 147803],  # Time step 3
        [214.12, 210.806, 214.42, 209.2, 2596288],  # Time step 4
        [214.46, 214.16, 214.53, 213.17, 1744757]   # Time step 5 (most recent)
    ])
    columns = ['Adj Close', 'Open', 'High', 'Low', 'Volume']
else:
    scaler = joblib.load(SCALER_CHANGE_VOLUME_FILE)
    # ['Change Adj Close', 'Volume']
    new_values = np.array([
        [0.0, 3618437],  # Time step 1 (oldest)
        [-0.07, 12186227],  # Time step 2
        [-3.26, 147803],  # Time step 3
        [3.32, 2596288],  # Time step 4
        [0.34, 1744757]   # Time step 5 (most recent)
    ])
    columns = ['Change Adj Close', 'Volume']

# Convert new values to DataFrame with the appropriate column names
new_values_df = pd.DataFrame(new_values, columns=columns)

# Normalize the new input values using the loaded scaler
normalized_new_values = scaler.transform(new_values_df)

# Check the normalized values
print(f"The normalized_new_values are: {normalized_new_values}")

# Reshape the input to match the model's expected input shape (batch_size, window_size, num_features)
num_features = len(columns)
normalized_new_values = normalized_new_values.reshape(1, -1, num_features)

# Load the trained model
model = tf.keras.models.load_model(MODEL_FILE)

# Predict the next value
normalized_predicted_value = model.predict(normalized_new_values)
print(f"The normalized_predicted_value predicted next value is: {normalized_predicted_value[0][0]}")

# Create a placeholder array with the same shape as the original data
placeholder = np.zeros((1, num_features))
# Fill the placeholder with the predicted value for the first column
placeholder[0, 0] = normalized_predicted_value[0, 0]

# Inverse transform the placeholder array to get the original scale
inverse_transformed = scaler.inverse_transform(placeholder)
# Extract the predicted value
predicted_value = inverse_transformed[0, 0]

# Output the predicted value
if use_all_features:
    print(f"The predicted next Adj Close value is: {predicted_value}")
else:
    print(f"The predicted next change in Adj Close is: {predicted_value}")


#20250410_model_v5.h5   Change in Adjusted Close Predictor
