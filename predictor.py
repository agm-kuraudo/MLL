import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import json
import os

# Set display options to show all rows and columns
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)

# Determine the current directory of the script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the path to the config file in the config folder
config_file_path = os.path.join(current_directory, 'config', 'predictor_up_down.json')

# Read the configuration from the JSON file
with open(config_file_path, 'r') as config_file:
    config_data = json.load(config_file)

# Access the values from the configuration
use_all_features = config_data["use_all_features"]
predict_direction = config_data["predict_direction"]
columns_all_features = config_data["columns_all_features"]
columns_change_volume = config_data["columns_change_volume"]
combined_data_path = config_data["combined_data_path"]
ticker = config_data["ticker"]
MODEL_FILE = config_data["MODEL_FILE"]
# Load the combined data CSV file
combined_data = pd.read_csv(combined_data_path)

# Print the loaded configuration and data
print("Configuration Loaded:")
print(json.dumps(config_data, indent=4))
print("\nCombined Data:")
print(combined_data.head())

do_the_rest_temp = False

# File paths for scaler
SCALER_ALL_FEATURES_FILE = "/app/models/scaler_all_features.pkl"
SCALER_CHANGE_VOLUME_FILE = "/app/models/scaler_change_volume.pkl"

print(tf.config.list_physical_devices('GPU'))

# Fit and save scaler for all features
scaler_all_features = MinMaxScaler()

scaler_all_features.fit(combined_data[columns_all_features])
joblib.dump(scaler_all_features, SCALER_ALL_FEATURES_FILE)

# Fit and save scaler for "Change Adj Close" and "Volume"
scaler_change_volume = MinMaxScaler()

combined_data['Change Adj Close'] = combined_data['Adj Close'].diff().fillna(0)
scaler_change_volume.fit(combined_data[columns_change_volume])
joblib.dump(scaler_change_volume, "/app/models/scaler_change_volume.pkl")

# Calculate the start and end dates
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=4)).strftime('%Y-%m-%d')
new_data = yf.download(ticker, start=start_date, end=end_date, interval='1h')

print(new_data.tail(5))

# Load the appropriate scaler
if use_all_features:
    scaler = joblib.load(SCALER_ALL_FEATURES_FILE)
    # ['Adj Close', 'Open', 'High', 'Low', 'Volume']
    new_values = new_data.tail(5).values.tolist()

    columns = ['Adj Close', 'Open', 'High', 'Low', 'Volume']
else:
    scaler = joblib.load(SCALER_CHANGE_VOLUME_FILE)
    # ['Change Adj Close', 'Volume']

    # Specify the columns you want to copy
    columns_to_copy = [('Close', ticker), ('Volume', ticker)]

    # Get the last 5 rows of the specified columns
    last_5_rows = new_data[columns_to_copy].tail(5)

    # Create a list for each of the last 5 rows
    new_values = last_5_rows.values.tolist()

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