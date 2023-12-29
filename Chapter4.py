import tensorflow_datasets as tfds

mnist_test, info = tfds.load(name='mnist', with_info=True)
print(info)
