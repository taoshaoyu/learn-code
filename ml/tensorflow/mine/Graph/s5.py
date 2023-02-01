# 简单程序，测试目标：
# 1. keras 创建模型
# 2. tf.function + concreteFunction + keras model + print node + tensorboard  对比 Graph

import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

@tf.function
def traceme(x):
    return model(x)

#print the node
for n in traceme.get_concrete_function(tf.zeros((1, 28, 28, 1))).graph.as_graph_def().node:
    print(f"{n.input}->{n.op}->{n.name}")

logdir = "logs/s5"
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=True)
# Forward pass
traceme(tf.zeros((1, 28, 28, 1)))
with writer.as_default():
    tf.summary.trace_export(name="s5", step=0, profiler_outdir=logdir)