from keras import layers
from keras import ops
import keras
from datasets.glove import embedding_matrix

@keras.saving.register_keras_serializable()
class PregeneratedEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, trainable=True, dtype=None):
        super().__init__(trainable=trainable, dtype=dtype)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, trainable=True)
        self.token_emb.build((1,))
        self.token_emb.set_weights([embedding_matrix])
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions