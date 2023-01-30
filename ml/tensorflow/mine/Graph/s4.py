import tensorflow as tf

@tf.function
def foo(x):
  y=x+1
  z=tf.multiply(y,2)
  return z


a=foo.get_concrete_function(1)

for node in a.graph.as_graph_def().node:
  print(f'{node.input} -> {node.op} -> {node.name}')

print("====================")

b=foo.get_concrete_function(-1)

for node in b.graph.as_graph_def().node:
  print(f'{node.input} -> {node.op} -> {node.name}')