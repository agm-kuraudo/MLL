import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch

# Check for GPU availability
print(tf.config.list_physical_devices('GPU'))

# Read the CSV file and skip the first two rows
df = pd.read_csv('/app/data/normalized_combined_data.csv', skiprows=2, header=None)
df.columns = ['Adj Close', 'Date', 'Ticker']

# Filter the DataFrame to only include 'Adj Close' column
filtered_df = df[['Adj Close']]
filtered_df.dropna(inplace=True)
print(filtered_df.isna().sum())
print("---------------------------------------------")
print(filtered_df.describe())

# Convert the DataFrame to a numpy array
data = filtered_df['Adj Close'].values

# Split the data into training and validation sets without shuffling
split_time = int(len(data) * 0.8)
train_data = data[:split_time]
val_data = data[split_time:]
print(f"Training data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")

# Reshape the data to ensure it has the correct shape
train_data = train_data.reshape(-1, 1)
val_data = val_data.reshape(-1, 1)
print(f"Reshaped training data shape: {train_data.shape}")
print(f"Reshaped validation data shape: {val_data.shape}")

# Convert the training data to a TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
print("Converted training and validation data to TensorFlow datasets.")

# Define the window size
window_size = 5

# Function to create windowed dataset
def create_windowed_dataset(dataset):
    windowed_dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    windowed_dataset = windowed_dataset.flat_map(lambda window: window.batch(window_size + 1))
    windowed_dataset = windowed_dataset.map(lambda window: (tf.reshape(window[:-1], (window_size, 1)), tf.reshape(window[-1], (1,))))
    return windowed_dataset.batch(32).prefetch(1)  # Batch and prefetch the dataset

# Create windowed datasets for training and validation
train_windowed_dataset = create_windowed_dataset(train_dataset)
val_windowed_dataset = create_windowed_dataset(val_dataset)
print("Windowed datasets created for training and validation.")

# Verify the shape of the windowed data
for x, y in train_windowed_dataset.take(1):
    print(f"Windowed input shape: {x.shape}, Windowed target shape: {y.shape}")

def build_model(hp):
    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(window_size, 1)),
        tf.keras.layers.SimpleRNN(100, return_sequences=True),
        tf.keras.layers.SimpleRNN(100),
        # tf.keras.layers.Conv1D(filters=hp.Int('units', min_value=128, max_value=256, step=64),
        #                        kernel_size=hp.Int('kernels', min_value=3, max_value=9, step=3),
        #                        strides=hp.Int('strides', min_value=1, max_value=3, step=1),
        #                        padding="causal", activation=tf.nn.relu),
        #tf.keras.layers.Dense(hp.Int('units', min_value=10, max_value=40, step=2), activation="relu"),
        # tf.keras.layers.Dense(10, activation="relu"),
        # tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    print("Model defined.")
    #model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(momentum=hp.Choice('momentum', values=[.9,.7,.5,.3]), learning_rate=1e-4))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(momentum=0.9,
                                                                learning_rate=1e-4))
    print("Model compiled.")
    return model


# tuner = RandomSearch(build_model, objective='loss', max_trials=150, executions_per_trial=1, directory='models', project_name='trade_model')
#
# tuner.search(train_windowed_dataset, epochs=20, validation_data=val_windowed_dataset, verbose=1)
#
# #lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))
#
# tuner.results_summary()

# model = tuner.get_best_models(num_models=1)[0]
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(window_size, 1)),
    tf.keras.layers.SimpleRNN(100, return_sequences=True),
    tf.keras.layers.SimpleRNN(100),
    # tf.keras.layers.Conv1D(filters=hp.Int('units', min_value=128, max_value=256, step=64),
    #                        kernel_size=hp.Int('kernels', min_value=3, max_value=9, step=3),
    #                        strides=hp.Int('strides', min_value=1, max_value=3, step=1),
    #                        padding="causal", activation=tf.nn.relu),
    #tf.keras.layers.Dense(hp.Int('units', min_value=10, max_value=40, step=2), activation="relu"),
    # tf.keras.layers.Dense(10, activation="relu"),
    # tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(momentum=0.9,
                                                            learning_rate=1e-4))
# Train the model with validation data
#history = model.fit(train_windowed_dataset, epochs=100, callbacks=[lr_schedule], validation_data=val_windowed_dataset, verbose=1)
history = model.fit(train_windowed_dataset, epochs=100, validation_data=val_windowed_dataset, verbose=1)
print("Model training completed.")

# Save the model to a file
model.save('/app/models/my_model_v4.h5')
print("Model saved successfully.")

# Plot training and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Training and Validation loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/app/tmp/training_validation_loss.png')
#plt.show()

# lrs=1e-8 * (10 ** (np.arange(100) / 20))
# plt.semilogy(lrs, history.history['loss'])
# plt.axis([1e-8, 1e-3, 0, 300])
# plt.savefig('/app/tmp/learning_rate_tuning.png')


# Plot training and validation MAE if available
if 'mae' in history_dict:
    mae = history_dict['mae']
    val_mae = history_dict['val_mae']
    plt.plot(epochs, mae, 'go-', label='Training MAE')
    plt.plot(epochs, val_mae, 'mo-', label='Validation MAE')
    plt.title('Training and Validation MAE over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig('/app/tmp/training_validation_mae.png')
    plt.show()