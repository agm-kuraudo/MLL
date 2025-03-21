import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch
from datetime import datetime

class TimeSeriesModel:
    def __init__(self, data_file, model_output_file, window_size=5):
        self.data_file = data_file
        self.model_output_file = model_output_file
        self.window_size = window_size
        self.train_data = None
        self.val_data = None
        self.train_windowed_dataset = None
        self.val_windowed_dataset = None
        self.model = None
        self.history = None

    def check_gpu(self):
        print(tf.config.list_physical_devices('GPU'))

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.data_file, skiprows=2, header=None)
        df.columns = ['Adj Close', 'Open', 'High', 'Low', 'Volume', 'Date', 'Ticker']
        filtered_df = df[['Adj Close', 'Open', 'High', 'Low', 'Volume']]
        filtered_df.dropna(inplace=True)
        print(filtered_df.isna().sum())
        print("---------------------------------------------")
        print(filtered_df.describe())
        data = filtered_df.values.astype(np.float32)  # Ensure data is float32
        split_time = int(len(data) * 0.8)
        self.train_data = data[:split_time]
        self.val_data = data[split_time:]
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Validation data shape: {self.val_data.shape}")

    def create_windowed_dataset(self, dataset):
        windowed_dataset = dataset.window(self.window_size + 1, shift=1, drop_remainder=True)
        windowed_dataset = windowed_dataset.flat_map(lambda window: window.batch(self.window_size + 1))
        windowed_dataset = windowed_dataset.map(lambda window: (tf.reshape(window[:-1], (self.window_size, 5)), tf.reshape(window[-1, 0], (1,))))
        return windowed_dataset.batch(32).prefetch(1)

    def prepare_datasets(self):
        train_dataset = tf.data.Dataset.from_tensor_slices(self.train_data)
        val_dataset = tf.data.Dataset.from_tensor_slices(self.val_data)
        self.train_windowed_dataset = self.create_windowed_dataset(train_dataset)
        self.val_windowed_dataset = self.create_windowed_dataset(val_dataset)
        print("Windowed datasets created for training and validation.")

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.window_size, 5)),
            tf.keras.layers.SimpleRNN(100, return_sequences=True),
            tf.keras.layers.SimpleRNN(100),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(momentum=0.9, learning_rate=1e-4))
        print("Model defined and compiled.")

    def train_model(self, epochs=100):
        self.history = self.model.fit(self.train_windowed_dataset, epochs=epochs, validation_data=self.val_windowed_dataset, verbose=1)
        print("Model training completed.")

    def save_model(self):
        self.model.save(self.model_output_file)
        print("Model saved successfully.")

    def plot_loss(self):
        history_dict = self.history.history
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
        # plt.show()

    def plot_mae(self):
        history_dict = self.history.history
        if 'mae' in history_dict:
            mae = history_dict['mae']
            val_mae = history_dict['val_mae']
            epochs = range(1, len(mae) + 1)
            plt.plot(epochs, mae, 'go-', label='Training MAE')
            plt.plot(epochs, val_mae, 'mo-', label='Validation MAE')
            plt.title('Training and Validation MAE over epochs')
            plt.xlabel('Epochs')
            plt.ylabel('MAE')
            plt.legend()
            plt.savefig('/app/tmp/training_validation_mae.png')
            plt.show()

# Usage
data_file = "/app/data/new_normalized_combined_data.csv"
model_output_file = f'/app/models/{datetime.now().strftime("%Y%m%d")}_model_v5.h5'
ts_model = TimeSeriesModel(data_file, model_output_file)

ts_model.check_gpu()
ts_model.load_and_preprocess_data()
ts_model.prepare_datasets()
ts_model.build_model()
ts_model.train_model(epochs=100)
ts_model.save_model()
ts_model.plot_loss()
ts_model.plot_mae()