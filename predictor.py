import pandas as pd
import tensorflow as tf

# Read the CSV file and skip the first two rows
df = pd.read_csv('data/apple_hourly_data.csv', skiprows=2, header=None)

# Manually set the correct header
df.columns = ['Datetime', 'Adj Close', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Volume']

# Drop the first row which contains the header information
df = df.drop(0)

# Reset the index
df.reset_index(drop=True, inplace=True)

# Filter the DataFrame to only include 'Adj Close' column
filtered_df = df[['Adj Close']]

# Convert the DataFrame to a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices(filtered_df['Adj Close'].values)

# Define the window size
window_size = 3

# Create a windowed dataset
windowed_dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

# Flatten the windows into batches
def flatten_fn(window):
    return window.batch(window_size + 1)

windowed_dataset = windowed_dataset.flat_map(flatten_fn)

# Split the windows into features and labels
def split_features_labels(window):
    features = window[:-1]
    label = window[-1]
    return features, label

windowed_dataset = windowed_dataset.map(split_features_labels)

# Print the first few windows with features and labels
for features, label in windowed_dataset.take(5):
    print("Features:", features.numpy())
    print("Label:", label.numpy())