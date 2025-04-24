import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

# Flag to choose feature set
use_all_features = True  # Set to True for all features, False for "Change Adj Close" and "Volume"
predict_direction = True # Set to True for UP/DOWN prediction, False for next close price



print(tf.config.list_physical_devices('GPU'))
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



# File paths for scaler
SCALER_ALL_FEATURES_FILE = "/app/models/scaler_all_features.pkl"
SCALER_CHANGE_VOLUME_FILE = "/app/models/scaler_change_volume.pkl"
#MODEL_FILE = "/app/models/20250422_model_v5.h5" #Price change only model trained on "agm-karaudo/ml_trader_image_10"
#MODEL_FILE = "/app/models/20250422_model_ten_steps.h5" #Price change only - window size ten - model trained on "agm-karaudo/ml_trader_image_10"
MODEL_FILE = "/app/models/20250424_up_down.h5" #all features, up / down binary classifier - window size five only ten epochs - pretty rubbish at moment - model trained on "agm-karaudo/ml_trader_image_10"


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

# Output the prediction based on the type of prediction
if predict_direction:
    # For UP/DOWN prediction
    confidence_up = normalized_predicted_value[0][0]
    confidence_down = 1 - confidence_up
    predicted_label = 'UP' if confidence_up > 0.5 else 'DOWN'
    print(f"The predicted direction is: {predicted_label}")
    print(f"Confidence: UP = {confidence_up:.2f}, DOWN = {confidence_down:.2f}")
else:
    # For next close price prediction
    # Create a placeholder array with the same shape as the original data
    placeholder = np.zeros((1, num_features))
    # Fill the placeholder with the predicted value for the first column
    placeholder[0, 0] = normalized_predicted_value[0][0]

    # Inverse transform the placeholder array to get the original scale
    inverse_transformed = scaler.inverse_transform(placeholder)
    # Extract the predicted value
    predicted_value = inverse_transformed[0, 0]

    print(f"The predicted next change in Adj Close is: {predicted_value}")
