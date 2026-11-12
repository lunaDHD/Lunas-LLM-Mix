# Data
BATCH_SIZE = 64
MIN_STRING_LEN = 20  # Strings shorter than this will be discarded
MAX_STRING_LEN = 80 # Strings longer than this will be discarded
SEQ_LEN = 128  # Length of training sequences, in tokens

# Model
EMBED_DIM = 50
FEED_FORWARD_DIM = 128
NUM_HEADS = 2
NUM_LAYERS = 1
VOCAB_SIZE = 5002  # Limits parameters in model.

# Training
EPOCHS = 5
# Inference
NUM_TOKENS_TO_GENERATE = 80