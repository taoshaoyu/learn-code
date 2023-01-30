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

for n in traceme.get_concrete_function(tf.zeros((1, 28, 28, 1))).graph.as_graph_def().node:
    print(f"{n.input}->{n.op}->{n.name}")