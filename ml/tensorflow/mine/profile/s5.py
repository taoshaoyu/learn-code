"""
code come from: https://wiki.ncsa.illinois.edu/display/ISL20/Profile+Tensorflow+using+Tensorboard
用于演示 tensorboard_plugin 的使用
问题在于：根据测试，这个方法仅仅对 model.fit() 过程有效
"""

from datetime import datetime
import os
import tensorflow as tf
 
# The function to be traced.
@tf.function
def my_func(x):
  if x>0:
    return tf.multiply(x,2) 
  else:
    return -x


w = tf.Variable(tf.ones(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))
predict = tf.nn.softmax(tf.matmul(x, w) + b)

# Set up logging.
logdir = 'logs/profile/s5/'
writer = tf.summary.create_file_writer(logdir)

tf.summary.trace_on(graph=True, profiler=True)

z1 = my_func(1)
z2 = my_func(-1)
with writer.as_default():
  tf.summary.trace_export(
      name="s5",
      step=0,
      profiler_outdir=logdir)
          