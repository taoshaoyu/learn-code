import tensorflow as tf

NUM_EXAMPLES = 201

x = tf.linspace(-2,2, NUM_EXAMPLES)
x = tf.cast(x, tf.float32)

def f(x):
  return x * 3.0 + 2.0

noise = tf.random.normal(shape=[NUM_EXAMPLES])

y = f(x) + noise

class MyModelKeras(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be randomly initialized
    self.w = tf.Variable(5.0)
    self.b = tf.Variable(0.0)

  def call(self, x):
    return self.w * x + self.b

keras_model = MyModelKeras()

keras_model.compile(
    run_eagerly=False,
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss=tf.keras.losses.mean_squared_error,
)

with tf.profiler.experimental.Profile("logs/mine/simple/simle-2/"):
  keras_model.fit(x, y, epochs=10, batch_size=1000)
