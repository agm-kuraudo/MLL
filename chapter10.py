import tensorflow as tf

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))

for x, y in dataset:
    print(x.numpy(), y.numpy())

    # [0 1 2 3][4]
    # [1 2 3 4][5]
    # [2 3 4 5][6]
    # [3 4 5 6][7]
    # [4 5 6 7][8]
    # [5 6 7 8][9]


#import keras
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
import tensorflow as tf
from tensorflow import keras
# import time

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level





time_1 = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time_1, .05)
baseline = 10
amplitude = 15
slope = 0.09
noise_level = 6

print("Hello")
# Create the series
series = baseline + trend(time_1, slope) + seasonality(time_1, period=365, amplitude=amplitude)
# Update with noise
series += noise(time_1, noise_level, seed=42)

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


split_time = 1000
time_train = time_1 [:split_time]
x_train = series[:split_time]
time_valid = time_1[split_time:]
x_valid = series[split_time:]
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
