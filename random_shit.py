import keras
import numpy as np

from config import *
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "torch" or "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from datasets.simplebooks import vocab
from models.PregeneratedEmbedding import PregeneratedEmbedding
from models.TransformerBlock import TransformerBlock

def cosineSimilarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    model = keras.saving.load_model("./exports/V0.2.keras")
    model.summary()
    embeddings = model.get_layer(index=True)
    def cosSim(a, b):
        return cosineSimilarity(keras.ops.reshape(embeddings.token_emb(np.array(a)), -1), keras.ops.reshape(embeddings.token_emb(np.array(b)), -1))
    count = 0
    index_to_word = {}
    word_to_index = {}
    for index, word in enumerate(vocab):
        index_to_word[index] = word
        word_to_index[word] = index
    for i in range(0, VOCAB_SIZE):
        sim = cosSim(i, 1)
        if (sim > 0.70):
            count += 1
            print(index_to_word[i])
    print(cosSim(word_to_index["woman"] - word_to_index["man"] + word_to_index["king"], word_to_index["queen"]))
    print(embeddings)
    print(embeddings.token_emb)
    print(count)

if __name__ == "__main__":
    main()