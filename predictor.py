import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

SCALER_FILE = "/app/models/new_scaler.pkl"

# Load the scaler
scaler = joblib.load(SCALER_FILE)


#['Adj Close', 'Open', 'High', 'Low', 'Volume'
new_values = np.array([
    [214.13, 212.35, 214.76, 212.35, 3618437],  # Time step 1 (oldest)
    [214.06, 214.11, 214.29, 213.67, 12186227],  # Time step 2
    [210.8, 212.18, 214.07, 210.76, 147803],  # Time step 3
    [214.12, 210.806, 214.42, 209.2, 2596288],  # Time step 4
    [214.46, 214.16, 214.53, 213.17, 1744757]   # Time step 5 (most recent)
])

# Convert new values to DataFrame with the appropriate column names
new_values_df = pd.DataFrame(new_values, columns=['Adj Close', 'Open', 'High', 'Low', 'Volume'])

# Normalize the new input values using the loaded scaler
normalized_new_values = scaler.transform(new_values_df)

# Check the normalized values
print(f"The normalized_new_values are: {normalized_new_values}")

# Reshape the input to match the model's expected input shape (batch_size, window_size, num_features)
normalized_new_values = normalized_new_values.reshape(1, -1, 5)

# Load the trained model
model = tf.keras.models.load_model('/app/models/20250321_model_v5.h5')

# Predict the next value
normalized_predicted_value = model.predict(normalized_new_values)
print(f"The normalized_predicted_value predicted next value is: {normalized_predicted_value[0][0]}")

# Create a placeholder array with the same shape as the original data
placeholder = np.zeros((1, 5))
# Fill the placeholder with the predicted value for the 'Adj Close' column
placeholder[0, 0] = normalized_predicted_value[0, 0]

# Inverse transform the placeholder array to get the original scale
inverse_transformed = scaler.inverse_transform(placeholder)
# Extract the predicted 'Adj Close' value
predicted_value = inverse_transformed[0, 0]

# Output the predicted value
print(f"The predicted next value is: {predicted_value}")