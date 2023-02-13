"""
code come from: https://wiki.ncsa.illinois.edu/display/ISL20/Profile+Tensorflow+using+Tensorboard
仅仅用来观察 onednn_Verbose的信息

"""

from datetime import datetime
import os
import tensorflow as tf
 
# The function to be traced.
@tf.function
def my_func(x, y):
  # A simple hand-rolled layer.
  return tf.nn.relu(tf.matmul(x, y))

# Sample data for your function.
x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

z = my_func(x, y)
