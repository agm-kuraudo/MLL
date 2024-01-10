
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import multiprocessing


def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                                      input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


train_data = tfds.load('cats_vs_dogs', split='train', with_info=True)

file_pattern = f'/root/tensorflow_datasets/cats_vs_dogs/4.0.1/cats_vs_dogs-train.tfrecord*'
files = tf.data.Dataset.list_files(file_pattern)

train_dataset = files.interleave(tf.data.TFRecordDataset,
                                 cycle_length=4,  # This specifies input elements processed concurrently
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)  # number of parallel calls -
# should be same as CPU count


def read_tfrecord(serialized_example):
    feature_description = {
        "image": tf.io.FixedLenFeature((), tf.string, ""),
        "label": tf.io.FixedLenFeature((), tf.int64, -1)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (300, 300))
    return image, example['label']


cores = multiprocessing.cpu_count()
print(cores)
train_dataset = train_dataset.map(read_tfrecord, num_parallel_calls=cores) # this makes the T bit parallel
#train_dataset = train_dataset.cache()

train_dataset = train_dataset.shuffle(1024).batch(32)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

model = create_model()
model.fit(train_dataset, epochs=10, verbose=1)
