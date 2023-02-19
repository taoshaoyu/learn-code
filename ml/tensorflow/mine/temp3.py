import tensorflow.compat.v1 as tf

# 定义两个const tensor
g=tf.Graph()
with g.as_default():
    m1 = tf.constant([[2, 2]])
    m2 = tf.constant([[3], [3]])
    dot_operation = tf.matmul(m1, m2)
    print(dot_operation)
    print(m1*2)
    
with tf.Session(g) as sess:
    result_ = sess.run(dot_operation)
    print(result_)

