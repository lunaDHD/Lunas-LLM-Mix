from config import *
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "torch" or "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from datasets.simplebooks import vocab
from models.BaseGPT import CompileBaseGPT
from models.TextGenerator import getGenCallback, genTokens

def main():
    model = CompileBaseGPT(MAX_STRING_LEN, VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FEED_FORWARD_DIM, "adam")
    model.summary()
    genTokens(vocab, model, MAX_STRING_LEN, "Hi")

if __name__ == "__main__":
    main()