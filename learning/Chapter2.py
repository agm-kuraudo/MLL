import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#A handy shortcut for accessing data built into Keras
data = tf.keras.datasets.fashion_mnist

#Call its load_data method to return our traning and test sets
(training_images, training_labels), (test_images, test_labels) = data.load_data()

#This seems odd at first... what we are doing is make sure every pixel has a value between 0 and 1 which normalises the data
training_images = training_images / 255.0
test_images = test_images / 255.0

#Building the model as discussed above
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#Specificy the loss function and optimiser  adam is an enhanced version of the 'sgd' we used previously
#The loss function is a good one to use with category analysis
#We specify the metrics to provide a view of how accurate the model is with regard to its guesses.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Run the training program for 5 epochs
model.fit(training_images, training_labels, epochs=5)

#We can now do something new... evaluate the model using test data!
model.evaluate(test_images, test_labels)
