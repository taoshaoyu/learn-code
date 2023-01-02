import tensorflow as tf


#with tf.profiler.experimental.Profile("logs/ss1/"):
#    for i in range(7):
#        o = tf.fill([3,3], 10)

#tf.zeros([1,1])
#with tf.profiler.experimental.Profile("logs/ss3/"):
#    for i in range(7):
#        tf.zeros([1,1])

#tf.constant([1, 2, 3, 4, 5, 6])
#with tf.profiler.experimental.Profile("logs/ss4/"):
#    for i in range(7):
#        tf.constant([1, 2, 3, 4, 5, 6])

tf.constant([1, 2, 3, 4, 5, 6])
with tf.profiler.experimental.Profile("logs/ss5/"):
    tf.constant([1, 2, 3, 4, 5, 6])
    tf.constant([1, 2, 3, 4, 5, 6])
    tf.constant([1, 2, 3, 4, 5, 6])
    tf.constant([1, 2, 3, 4, 5, 6])
    tf.constant([1, 2, 3, 4, 5, 6])