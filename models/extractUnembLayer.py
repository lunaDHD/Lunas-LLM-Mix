import keras
from config import VOCAB_SIZE, EMBED_DIM
UnembModel = keras.models.load_model("./exports/Embed.keras")
UnembWeights = UnembModel.get_layer(index=3).get_weights()
print(UnembModel.get_layer(index=3))
UnembLayer = keras.layers.Dense(VOCAB_SIZE, trainable=True)
UnembLayer.build((EMBED_DIM,))
UnembLayer.set_weights(UnembWeights)
