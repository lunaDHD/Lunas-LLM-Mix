import numpy as np
from datasets.simplebooks import vocab
from config import EMBED_DIM

path_to_glove_file = f"./datasets/Glove/glove.6B.{EMBED_DIM}d.txt"

word_index = dict(zip(vocab, range(len(vocab))))

embeddings_index = {}
with open(path_to_glove_file, encoding='utf-8') as f:
    for line in f.readlines():
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

num_tokens = len(vocab) + 1
embedding_dim = EMBED_DIM
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))