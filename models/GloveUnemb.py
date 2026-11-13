import keras
from keras import layers
from models.PregeneratedEmbedding import PregeneratedEmbedding

def CompileGlove(MAX_LEN, VOCAB_SIZE, EMBED_DIM, OPTIM):
    inputs = layers.Input(shape=(MAX_LEN,), dtype="int32")
    embedding_layer = PregeneratedEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM, trainable=True)
    x = embedding_layer(inputs)
    noise_layer = layers.GaussianNoise(0.01)
    x = noise_layer(x, training=True)
    outputs = layers.Dense(VOCAB_SIZE)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        OPTIM,
        loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model