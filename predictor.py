import pandas as pd
import tensorflow as tf

# Read the CSV file and skip the first two rows
df = pd.read_csv('/app/data/apple_hourly_data.csv', skiprows=2, header=None)
print("CSV file read successfully.")

# Manually set the correct header
df.columns = ['Datetime', 'Adj Close', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Volume']
print("Headers set successfully.")

# Drop the first row which contains the header information
df = df.drop(0)
print("First row dropped.")

# Reset the index
df.reset_index(drop=True, inplace=True)
print("Index reset.")

# Filter the DataFrame to only include 'Adj Close' column
filtered_df = df[['Adj Close']]
print(f"Filtered DataFrame shape: {filtered_df.shape}")

# Convert the DataFrame to a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices(filtered_df['Adj Close'].values)
print("Converted DataFrame to TensorFlow dataset.")

# Define the window size
window_size = 5

# Create a windowed dataset
windowed_dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
print("Windowed dataset created.")

# Flatten the windows into batches
def flatten_fn(window):
    return window.batch(window_size + 1)

windowed_dataset = windowed_dataset.flat_map(flatten_fn)
print("Windows flattened into batches.")

# Split the windows into features and labels
def split_features_labels(window):
    features = window[:-1]
    label = window[-1]
    return features, label

windowed_dataset = windowed_dataset.map(split_features_labels)
print("Windows split into features and labels.")

# Reshape the features to match the input shape expected by the model
windowed_dataset = windowed_dataset.map(lambda features, label: (tf.expand_dims(features, axis=-1), label))
print("Features reshaped.")

# Batch the dataset
batch_size = 32
windowed_dataset = windowed_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
print("Dataset batched and prefetched.")

# Repeat the dataset to ensure it doesn't run out of data
# windowed_dataset = windowed_dataset.repeat()
# print("Dataset repeated.")

# Calculate the number of steps per epoch
total_windows = len(filtered_df) - window_size
steps_per_epoch = total_windows // batch_size
print(f"Total windows: {total_windows}, Steps per epoch: {steps_per_epoch}")

# Print the first few windows with features and labels
# for features, label in windowed_dataset.take(5):
#     print("Features:", features.numpy())
#     print("Label:", label.numpy())

# Count the total number of batches
# Remove the repeat() call for counting batches
windowed_dataset_no_repeat = windowed_dataset.unbatch().batch(batch_size)

# Limit the number of iterations to avoid infinite loop
max_batches = 3000  # Set a reasonable limit for counting
batch_count = 0

for batch in windowed_dataset_no_repeat:
    batch_count += 1
    print(f"Batch {batch_count}:")
    if batch_count >= max_batches:
        break

print(f"Total number of batches (up to {max_batches}): {batch_count}")

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(window_size, 1)),  # Adjust the input shape here
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="causal", activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])
print("Model defined.")

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.5))
print("Model compiled.")

# Train the model
history = model.fit(windowed_dataset, epochs=100, steps_per_epoch=steps_per_epoch, verbose=1)
print("Model training completed.")

import matplotlib.pyplot as plt
history_dict = history.history
loss = history_dict['loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.title('Training loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/app/tmp/training_loss.png')
plt.show()


if 'mae' in history_dict:
    mae = history_dict['mae']
    plt.plot(epochs, mae, 'go-', label='Training MAE')
    plt.title('Training MAE over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig('/app/tmp/training_mae.png')