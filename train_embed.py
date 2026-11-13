from config import *
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "torch" or "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

IS_EMBED = True
from datasets.simplebooks import vocab, text_ds
from models.GloveUnemb import CompileGlove
from models.TextGenerator import getGenCallback
import keras

def main():
    #model = CompileGlove(MAX_STRING_LEN, VOCAB_SIZE, EMBED_DIM, "adam")
    model = keras.models.load_model("./exports/Embed.keras") 
    model.summary()
    callback = getGenCallback(vocab)
    model.fit(text_ds, verbose=2, epochs=EPOCHS, callbacks=[callback])
    keras.models.save_model(model, "./exports/Embed.keras") 

if __name__ == "__main__":
    main()