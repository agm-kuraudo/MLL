import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Billy big bollocks has big old bollocks'
]

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
# {'today': 1, 'is': 2, 'a': 3, 'day': 4, 'sunny': 5, 'rainy': 6}

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
# [[1, 2, 3, 5, 4], [1, 2, 3, 6, 4]]

padded = pad_sequences(sequences)
print(padded)
# [[ 0  0  2  3  4  8  5]
#  [ 0  0  2  3  4  9  5]
#  [10  6  7 11  6 12  7]]
