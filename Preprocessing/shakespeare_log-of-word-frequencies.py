import tensorflow as tf

path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read the file
text = None
with open(path_to_file, 'rb') as f:
    text = f.read().decode(encoding='utf-8')

# The number of characters in text (length)
print(f'Length of text: {len(text)} characters')

# The first 250 characters in text
print(text[:250])

