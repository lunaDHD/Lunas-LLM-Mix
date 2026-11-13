from config import *
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "torch" or "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

IS_EMBED = False
from models.TokenAndPositionEmbedding import TokenAndPositionEmbedding
from models.PregeneratedEmbedding import PregeneratedEmbedding
from models.TransformerBlock import TransformerBlock
from datasets.simplebooks import vocab
from models.TextGenerator import getGenCallback, genTokens
import keras

def main():
    model = keras.saving.load_model("./exports/V0.1.keras")
    model.summary()
    while True:
        startTokens = input("> ")
        genTokens(vocab, model, MAX_STRING_LEN, startTokens)

if __name__ == "__main__":
    main()