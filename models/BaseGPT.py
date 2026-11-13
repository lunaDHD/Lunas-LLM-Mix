import keras
from keras import layers
from models.PregeneratedEmbedding import PregeneratedEmbedding
from models.TokenAndPositionEmbedding import TokenAndPositionEmbedding
from models.TransformerBlock import TransformerBlock

def CompileBaseGPT(MAX_LEN, VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FEED_FORWARD_DIM, OPTIM, pregeneratedEmb=True, pregeneratedUnemb=None):
    inputs = layers.Input(shape=(MAX_LEN,), dtype="int32")
    embedding_layer = None
    if pregeneratedEmb:
        embedding_layer = PregeneratedEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM)
    else:
        embedding_layer = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM)
    x = embedding_layer(inputs)
    for _ in range(NUM_LAYERS):
        transformer_block = TransformerBlock(EMBED_DIM, NUM_HEADS, FEED_FORWARD_DIM)
        x = transformer_block(x)
    if pregeneratedUnemb == None:
        outputs = layers.Dense(VOCAB_SIZE)(x)
    else:
        outputs = pregeneratedUnemb(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        OPTIM,
        loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model