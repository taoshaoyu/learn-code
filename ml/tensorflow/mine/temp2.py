#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf

# 定义两个const tensor
m1 = tf.constant([[2, 2]])
m2 = tf.constant([[3], [3]])
dot_operation = tf.matmul(m1, m2)



print(dot_operation)
print(m1*2)

g=tf.get_default_graph()
for node in g.as_graph_def().node:
  print(f'{node.input} -> {node.name}')