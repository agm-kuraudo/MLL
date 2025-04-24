import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch
from datetime import datetime
import json

class TimeSeriesModel:
    def __init__(self, data_file, model_output_file, window_size=5, use_all_features=True, predict_direction=False):
        self.data_file = data_file
        self.model_output_file = model_output_file
        self.window_size = window_size
        self.use_all_features = use_all_features
        self.predict_direction = predict_direction
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
        df['Date'] = pd.to_datetime(df['Date'])
        df['Ticker'] = df['Ticker'].astype('category').cat.codes
        df['Change Adj Close'] = df['Adj Close'].diff().fillna(0)
        df['Label'] = np.where(df['Change Adj Close'] > 0, 1, 0)  # 1 for UP, 0 for DOWN

        if self.use_all_features:
            filtered_df = df[
                ['Adj Close', 'Open', 'High', 'Low', 'Volume', 'Ticker', 'Label']] if self.predict_direction else df[
                ['Adj Close', 'Open', 'High', 'Low', 'Volume', 'Ticker']]
        else:
            filtered_df = df[['Change Adj Close', 'Volume', 'Ticker', 'Label']] if self.predict_direction else df[
                ['Change Adj Close', 'Volume', 'Ticker']]

        filtered_df.dropna(inplace=True)
        print(filtered_df.isna().sum())
        print("---------------------------------------------")
        print(filtered_df.describe())
        data = filtered_df.values.astype(np.float32)
        split_time = int(len(data) * 0.8)
        self.train_data = data[:split_time]
        self.val_data = data[split_time:]
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Validation data shape: {self.val_data.shape}")

    def create_windowed_dataset(self, dataset, window_size):
        def windowed_data_generator():
            for ticker, group in dataset.groupby('Ticker'):
                if self.use_all_features:
                    group_data = group[['Adj Close', 'Open', 'High', 'Low', 'Volume']].values
                else:
                    group_data = group[['Change Adj Close', 'Volume']].values

                labels = group['Label'].values if self.predict_direction else group['Adj Close'].values

                for i in range(len(group_data) - window_size):
                    window = group_data[i:i + window_size]
                    label = labels[i + window_size] if self.predict_direction else np.array([labels[i + window_size]])
                    yield (window, label)

        feature_count = 5 if self.use_all_features else 2
        output_signature = (
            tf.TensorSpec(shape=(window_size, feature_count), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32) if self.predict_direction else tf.TensorSpec(shape=(1,),
                                                                                                 dtype=tf.float32)
        )
        windowed_dataset = tf.data.Dataset.from_generator(
            windowed_data_generator,
            output_signature=output_signature
        )
        return windowed_dataset.batch(32).prefetch(1)

    def prepare_datasets(self, window_size):
        if self.use_all_features:
            columns = ['Adj Close', 'Open', 'High', 'Low', 'Volume', 'Ticker', 'Label'] if self.predict_direction else [
                'Adj Close', 'Open', 'High', 'Low', 'Volume', 'Ticker']
        else:
            columns = ['Change Adj Close', 'Volume', 'Ticker', 'Label'] if self.predict_direction else [
                'Change Adj Close', 'Volume', 'Ticker']

        df_train = pd.DataFrame(self.train_data, columns=columns)
        df_val = pd.DataFrame(self.val_data, columns=columns)
        self.train_windowed_dataset = self.create_windowed_dataset(df_train, window_size)
        self.val_windowed_dataset = self.create_windowed_dataset(df_val, window_size)
        print("Windowed datasets created for training and validation.")

    def build_model(self, hp=None):
        feature_count = 5 if self.use_all_features else 2

        # Load saved hyperparameters if available
        try:
            with open('/app/models/best_hyperparameters.json', 'r') as f:
                saved_hyperparameters = json.load(f)
                units = saved_hyperparameters['units']
                learning_rate = saved_hyperparameters['learning_rate']
                momentum = saved_hyperparameters['momentum']
        except FileNotFoundError:
            units = hp.Int('units', min_value=50, max_value=200, step=50)
            learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG')
            momentum = hp.Float('momentum', min_value=0.0, max_value=0.9, step=0.1)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.window_size, feature_count)))
        for i in range(hp.Int('num_layers', 1, 3) if hp else 2):  # Default to 2 layers if hp is None
            model.add(tf.keras.layers.GRU(units=units, return_sequences=True))
        model.add(tf.keras.layers.GRU(units=units))  # Ensure the last GRU layer does not return sequences
        model.add(tf.keras.layers.Dense(1, activation='sigmoid' if self.predict_direction else None))  # Single output
        loss = "binary_crossentropy" if self.predict_direction else "mse"
        metrics = ['accuracy'] if self.predict_direction else []
        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.SGD(momentum=momentum, learning_rate=learning_rate),
            metrics=metrics
        )
        return model

    def tune_hyperparameters(self):
        tuner = RandomSearch(
            self.build_model,
            objective='accuracy',
            max_trials=10,
            executions_per_trial=1,  # Set to 1 for faster tuning
            directory='my_dir',
            project_name='time_series_tuning'
        )
        tuner.search(self.train_windowed_dataset, epochs=5, validation_data=self.val_windowed_dataset)
        self.model = tuner.get_best_models(num_models=1)[0]
        print("Hyperparameter tuning completed.")

        # Save the best model
        self.model.save(self.model_output_file)
        print("Best model saved successfully.")

        # Save the best hyperparameters
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        hyperparameters_dict = {
            'units': best_hyperparameters.get('units'),
            'learning_rate': best_hyperparameters.get('learning_rate'),
            'momentum': best_hyperparameters.get('momentum')
        }
        with open('/app/models/best_hyperparameters.json', 'w') as f:
            json.dump(hyperparameters_dict, f)
        print("Best hyperparameters saved successfully.")

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
model_output_file = f'/app/models/{datetime.now().strftime("%Y%m%d")}_up_down.h5'
ts_model = TimeSeriesModel(data_file, model_output_file, use_all_features=True, window_size=5, predict_direction=True)

ts_model.check_gpu()
ts_model.load_and_preprocess_data()
ts_model.prepare_datasets(window_size=5)

# First run: tune hyperparameters
ts_model.tune_hyperparameters()


# Subsequent runs: use saved hyperparameters
ts_model.build_model()
ts_model.train_model(epochs=100)
ts_model.save_model()
ts_model.plot_loss()
ts_model.plot_mae()
