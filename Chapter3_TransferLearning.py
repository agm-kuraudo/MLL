import urllib.request
import zipfile
import tensorflow as tf

from keras.src.optimizers import RMSprop
import sys
import numpy as np

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Python {sys.version}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

import urllib.request
import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop

weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['acc'])






# Download the training data and stick it in a directory
url = "https://storage.googleapis.com/learning-datasets/horse-or-human.zip"
validation_url = "https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"

file_name = "horse-or-human.zip"
validation_file_name = "validation-horse-or-human.zip"

training_dir = "/app/horse-or-human/training/"
validation_dir = "/app/horse-or-human/validation/"

urllib.request.urlretrieve(url, file_name)

# zip_ref = zipfile.ZipFile(file_name, 'r')
# zip_ref.extractall(training_dir)
# zip_ref.close()
#
# urllib.request.urlretrieve(validation_url, validation_file_name)
# zip_ref = zipfile.ZipFile(validation_file_name, 'r')
# zip_ref.extractall(validation_dir)
# zip_ref.close()

# Create an instance of ImageDataGenerator, and we can generate images for training by flowing from a directory

# All images will be rescaled by 1./255
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    batch_size=20,
    target_size=(150, 150),
    class_mode='binary'  # This might be categorical if more than two labels
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    batch_size=20,
    target_size=(150, 150),
    class_mode='binary'  # This might be categorical if more than two labels
)


history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    verbose=1
)

test_images = ["Picture1.jpg", "Picture2.png"]


from keras.preprocessing import image

for pic in test_images:

    # Here we are loading the image (which can be any size) then resizes to expected side (150, 150).
    # The next line of code converts to a 2D array. The models needs a 3D array as specified by the "input shape".
    # Fortunately numpy provides an expand_dims method that handles this for us.
    img = image.load_img("/app/newtestdata/" + pic, target_size=(150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # this line stacks vertically to match our training data
    image_tensor = np.vstack([x])
    # this does the predicting
    classes = model.predict(image_tensor)
    print(classes)
    print(classes[0])

    if classes[0] > 0.5:
        print(pic + ' is a human')
    else:
        print(pic + ' is a horse')



