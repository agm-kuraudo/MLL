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
#matplotlib.use('TkAgg')
plt.figure(figsize=(10, 6))
plot_series(time_1, series)
#plt.savefig("/app/tmp/chart2.png")
plt.show()
#time.sleep(120)




split_time = 1000
time_train = time_1[:split_time]
x_train = series[:split_time]
time_valid = time_1[split_time:]
x_valid = series[split_time:]
plt.figure(figsize=(10, 6))
plot_series(time_train, x_train)
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plt.show()




naive_forecast = series[split_time - 1:-1]


plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, naive_forecast)


plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start=0, end=150)
plot_series(time_valid, naive_forecast, start=1, end=151)


m = keras.metrics.MeanSquaredError()
m.update_state(x_valid, naive_forecast)
print(m.result())

m = keras.metrics.MeanAbsoluteError()
m.update_state(x_valid, naive_forecast)
print(m.result())


def moving_average_forecast(series, window_size):
  """Forecasts the mean of the last few values.
     If window_size=1, then this is equivalent to naive forecast"""
  forecast = []
  for time in range(len(series) - window_size):
    forecast.append(series[time:time + window_size].mean())
  return np.array(forecast)


moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)
plt.show()

m = keras.metrics.MeanSquaredError()
m.update_state(x_valid, moving_avg)
print(m.result())

m = keras.metrics.MeanAbsoluteError()
m.update_state(x_valid, moving_avg)
print(m.result())



diff_series = (series[365:] - series[:-365])
diff_time = time_1[365:]


diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]

diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()

m = keras.metrics.MeanSquaredError()
m.update_state(x_valid, diff_moving_avg_plus_smooth_past)
print(m.result())

m = keras.metrics.MeanAbsoluteError()
m.update_state(x_valid, diff_moving_avg_plus_smooth_past)
print(m.result())
