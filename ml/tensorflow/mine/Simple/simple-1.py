import tensorflow as tf

NUM_EXAMPLES = 201

x = tf.linspace(-2,2, NUM_EXAMPLES)
x = tf.cast(x, tf.float32)

def f(x):
  return x * 3.0 + 2.0

noise = tf.random.normal(shape=[NUM_EXAMPLES])

y = f(x) + noise

class MyModel(tf.Module):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.w = tf.Variable(5.0)
    self.b = tf.Variable(0.0)

  def __call__(self, x):
    return self.w * x + self.b

def loss(target_y, predicted_y):
  return tf.reduce_mean(tf.square(target_y - predicted_y))

def train(model, x, y, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(y, model(x))

  dw, db = t.gradient(current_loss, [model.w, model.b])

  model.w.assign_sub(learning_rate * dw)
  model.b.assign_sub(learning_rate * db)

def training_loop(model, x, y):
  for _ in range(10):
    train(model, x, y, learning_rate=0.1)

model = MyModel()

training_loop(model, x, y)
