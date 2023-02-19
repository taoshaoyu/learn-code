import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# 定义两个const tensor
@tf.function
def foo(x):
    if x>0:
        return x*3
    else:
        return -x+3

print(foo(1))

