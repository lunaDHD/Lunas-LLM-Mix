from config import *
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "torch" or "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
IS_EMBED = True

from models.TokenAndPositionEmbedding import TokenAndPositionEmbedding
from models.PregeneratedEmbedding import PregeneratedEmbedding
from models.TransformerBlock import TransformerBlock
from datasets.simplebooks import vocab
from models.TextGenerator import embed_test
import keras
def main():
    model = keras.saving.load_model("./exports/Embed.keras")
    model.summary()
    while True:
        startTokens = input("> ")
        embed_test(vocab, model, MAX_STRING_LEN, startTokens)

if __name__ == "__main__":
    main()