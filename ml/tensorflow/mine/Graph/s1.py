# 简单程序，用来理解TF Graph

import tensorflow as tf

g1=tf.Graph()

with g1.as_default():
    a=tf.constant([1,2])
    b=tf.constant([1,1])
    result=a+b

for node in g1.as_graph_def().node:
  print(f'{node.input} -> {node.name}')