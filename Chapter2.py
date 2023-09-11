import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
