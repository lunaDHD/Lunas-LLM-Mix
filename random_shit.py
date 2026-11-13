import keras

from config import *
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "torch" or "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from models.PregeneratedEmbedding import PregeneratedEmbedding
from models.TransformerBlock import TransformerBlock

def main():
    model = keras.saving.load_model("./exports/Embed.keras")
    model.summary()

if __name__ == "__main__":
    main()