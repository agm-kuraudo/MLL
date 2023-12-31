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
    rotation_range=40,
    width_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode='binary'  # This might be categorical if more than two labels
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode='binary'  # This might be categorical if more than two labels
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator
)

test_images = ["Picture1.jpg", "Picture2.png"]

from keras.preprocessing import image

for pic in test_images:

    # Here we are loading the image (which can be any size) then resizes to expected side (300, 300).
    # The next line of code converts to a 2D array. The models needs a 3D array as specified by the "input shape".
    # Fortunately numpy provides an expand_dims method that handles this for us.
    img = image.load_img("/app/newtestdata/" + pic, target_size=(300,300))
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



