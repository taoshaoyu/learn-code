import tensorflow as tf

@tf.function
def foo(x):
  if x>0 :
    return tf.multiply(x,2) 
  else:
    return -x

print(foo(1))
print(foo(-1))

a=foo.get_concrete_function(1)

for node in a.graph.as_graph_def().node:
  print(f'{node.input} -> {node.op} -> {node.name}')

print("=====")
b=foo.get_concrete_function(-1)

for node in b.graph.as_graph_def().node:
  print(f'{node.input} -> {node.op} -> {node.name}')