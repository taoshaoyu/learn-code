# 学习使用 keras Model 的流程
# https://www.tensorflow.org/api_docs/python/tf/keras/Model
# 看上去netron 不支持MyModel 这种继承的类。。。

import numpy as np
import tensorflow as tf
from tensorflow import keras

# 1. 创建 keras.Model 对象
# 2. 提供给Model 对象以 Input / Output
inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(5, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="mean_squared_error")


model.save("my_h5_model_2.h5")
#print('My custom loss: ', model.loss_tracker.result().numpy())

import numpy as np
import tensorflow as tf
from tensorflow import keras

inputs = keras.Input(shape=(784,), name="fff")
outputs = keras.layers.Dense(10)(inputs)
outputs = keras.layers.Dense(10)(outputs)
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()
model.save("my_h5_model.h5")