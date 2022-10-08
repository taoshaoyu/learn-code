"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
"""
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.compat.v1.disable_eager_execution()

m1 = tf.constant([[2, 2]])
m2 = tf.constant([[3],
                  [3]])
dot_operation = tf.matmul(m1, m2)

print(dot_operation)  # wrong! no result

# method1 use session
sess = tf.compat.v1.Session()
result = sess.run(dot_operation)
print(result)
sess.close()

# method2 use session
with tf.compat.v1.Session() as sess:
    result_ = sess.run(dot_operation)
    print(result_)