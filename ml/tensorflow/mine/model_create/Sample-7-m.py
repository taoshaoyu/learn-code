# 学习自己做模型，和看模型
# https://www.tensorflow.org/guide/keras/save_and_serialize

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
#    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

model = get_model()

model.save("my_model_1")
model.save("my_model_1.h5")

model_2 = keras.Sequential([
    layers.Dense(512, input_shape=(3,)),
])
model_2.save("my_model_2")
model_2.save("my_model_2.h5")






