from config import *
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "torch" or "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

IS_EMBED = False
from datasets.simplebooks import vocab, text_ds
from models.BaseGPT import CompileBaseGPT
from models.TextGenerator import getGenCallback
from models.extractUnembLayer import UnembLayer
import keras

def main():
    model = CompileBaseGPT(MAX_STRING_LEN, VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FEED_FORWARD_DIM, "adam", pregeneratedUnemb=UnembLayer)
    #model = keras.models.load_model("./exports/V0.2.keras")
    model.summary()
    callback = getGenCallback(vocab)
    model.fit(text_ds, verbose=2, epochs=EPOCHS, callbacks=[callback])
    keras.models.save_model(model, "./exports/V0.2.keras")
 

if __name__ == "__main__":
    main()
