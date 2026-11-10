import keras
from keras import layers
from models.TokenAndPositionEmbedding import TokenAndPositionEmbedding
from models.TransformerBlock import TransformerBlock

def CompileBaseGPT(MAX_LEN, VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FEED_FORWARD_DIM, OPTIM):
    inputs = layers.Input(shape=(MAX_LEN,), dtype="int32")
    embedding_layer = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM)
    x = embedding_layer(inputs)
    for _ in range(NUM_LAYERS):
        transformer_block = TransformerBlock(EMBED_DIM, NUM_HEADS, FEED_FORWARD_DIM)
        x = transformer_block(x)
    outputs = layers.Dense(VOCAB_SIZE)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        OPTIM,
        loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model