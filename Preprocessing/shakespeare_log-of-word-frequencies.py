import tensorflow as tf
import string
import numpy as np
import matplotlib.pyplot as plt

path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

with open(path_to_file, 'rb') as f:
    text = f.read().decode(encoding='utf-8')
    print(f'Length of text: {len(text)} characters')
    print(text[:250])

apostrophe = ["'s"]
escapes = ["\'", "\n", "\r", "\t", "\b", "\f"]
numbers = [str(i) for i in range(10)]
waste_list = apostrophe + escapes + list(string.punctuation) + numbers

file_lines = []
words_count = {}
unique_words = set()

with open(path_to_file, "r") as f:
    for file_line in f:
        for waste in waste_list:
            file_line = file_line.replace(waste, "")
        file_line = file_line.lower()
        file_lines.append(file_line)
        for word in file_line.split():
            if word not in words_count:
                words_count[word] = 0
            words_count[word] += 1
            unique_words.add(word)

print("Processed file lines:\n", file_lines[:3], "...", file_lines[-3:])
print()
print("words_count:\n", words_count)
print()
print("unique_words:\n", unique_words)

counts = np.array(list(words_count.values()))
print("Counts:\n", counts)
print()
counts_log = np.log(counts)
print("Log counts:\n", counts_log)
print()
print("0:", len(counts_log) - np.count_nonzero(counts_log))
print()

plt.figure(figsize=(15, 10))
plt.hist(counts_log, bins=len(set(counts_log)))
plt.xlabel("Log of word frequency")
plt.ylabel("Count")
plt.title("Log of word frequency over Count", fontweight="bold")
plt.xticks(np.arange(min(counts_log), max(counts_log) + 0.2, 0.2))
plt.grid()
plt.show()

print(((0 <= counts_log) & (counts_log < 0.2)).sum())
print(((0.6 < counts_log) & (counts_log < 0.8)).sum())
print(((1 < counts_log) & (counts_log < 1.2)).sum())
print(((1.2 < counts_log) & (counts_log < 1.4)).sum())
print("...")
