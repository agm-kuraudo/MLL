import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# Load the scaler
scaler = joblib.load('/app/models/scaler.pkl')

# Define the new input values
new_values = np.array([229.92, 230.03, 231.18, 230.79, 230.65])

# Convert new values to DataFrame with the same column name
new_values_df = pd.DataFrame(new_values, columns=['Adj Close'])

# Normalize the new input values using the loaded scaler
normalized_new_values = scaler.transform(new_values_df)

# Check the normalized values
print(f"The normalized_new_values is: {normalized_new_values.flatten()}")

# Reshape the input to match the model's expected input shape (batch_size, window_size, 1)
normalized_new_values = normalized_new_values.reshape(1, -1, 1)

# Load the trained model
model = tf.keras.models.load_model('/app/models/my_model.h5')

# Predict the next value
normalized_predicted_value = model.predict(normalized_new_values)
print(f"The normalized_predicted_value predicted next value is: {normalized_predicted_value[0][0]}")

# Inverse transform the predicted value to get the original scale
predicted_value = scaler.inverse_transform(normalized_predicted_value.reshape(-1, 1)).flatten()

# Output the predicted value
print(f"The predicted next value is: {predicted_value[0]}")