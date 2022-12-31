"""
code come from: https://wiki.ncsa.illinois.edu/display/ISL20/Profile+Tensorflow+using+Tensorboard
尝试使用tensorflow.profiler.experimental.Profile + tensorboard 来观察 model.predict() 过程的性能
"""

from datetime import datetime
import os
import tensorflow 
 
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
 
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
 
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
 
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
 
# Create a TensorBoard callback
model.fit(train_images,
          train_labels,
          epochs=1,
          batch_size=128)

with tensorflow.profiler.experimental.Profile("logs/t/"):
    model.predict(test_images)
