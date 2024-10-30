import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow import keras
import time

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
#matplotlib.use('TkAgg')
plt.figure(figsize=(10, 6))
plot_series(time_1, series)
plt.savefig("/app/tmp/chart2.png")
#time.sleep(120)
