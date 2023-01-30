# Sample code, 用来理解 Sequential / Funcational Model
# https://becominghuman.ai/sequential-vs-functional-model-in-keras-20684f766057

import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential(
    [
        tf.keras.layers.Dense(2, activation="relu", name="layer1"),
        tf.keras.layers.Dense(3, activation="relu", name="layer2"),
        tf.keras.layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)

model.save("c2.h5")
model.summary()

# 如下代码跟上面的代码等价
# Create 3 layers
layer1 = tf.keras.layers.Dense(2, activation="relu", name="layer1")
layer2 = tf.keras.layers.Dense(3, activation="relu", name="layer2")
layer3 = tf.keras.layers.Dense(4, name="layer3")

# Call layers on a test input
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))