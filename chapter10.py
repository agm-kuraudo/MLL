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
