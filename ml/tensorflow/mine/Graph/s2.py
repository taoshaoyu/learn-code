import tensorflow as tf

def simple_relu(x):
  if tf.greater(x, 0):
    return x
  else:
    return 0
tf_simple_relu = tf.function(simple_relu)
a=tf_simple_relu.get_concrete_function(1)
for node in a.graph.as_graph_def().node:
  print(f'{node.input} -> {node.name}')