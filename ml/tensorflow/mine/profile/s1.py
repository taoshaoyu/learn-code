"""
code come from: https://wiki.ncsa.illinois.edu/display/ISL20/Profile+Tensorflow+using+Tensorboard

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
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
 
tboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '10,20')
 
model.fit(train_images,
          train_labels,
          epochs=10,
          batch_size=128,
          callbacks = [tboard_callback])
          