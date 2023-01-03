# TF1 + Mnist
# 一个老式的Mnist的范例，主要工作在于 函数 get_batch() 的实现
# https://juejin.cn/post/7086260725072527390#heading-0

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()
n_batchs = len(x_train) 

x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x') # 因为每张图片数据是 28*28=784 维的
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')  # 因为一共有 10 种类别的图片

w = tf.Variable(tf.ones(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))
predict = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

correct = tf.equal(tf.argmax(predict,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) # 计算准确率

def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    y_ret_data=[]
    for i in y_data[idxs]:
        tmp=np.zeros(10)
        tmp[i]=1
        y_ret_data.append(tmp)
    return x_data[idxs,:,:].reshape(batch_size,28*28), np.array(y_ret_data)

with tf.Session() as sess:
    sess.run(init)
    total_batch = 0
    last_batch = 0
    best = 0
    for epoch in range(100):
        for _ in range(n_batchs):
            # xx,yy = mnist.train.next_batch(batch_size)
            xx, yy = get_batch(x_train, y_train, batch_size=64)
            sess.run(opt, feed_dict={x:xx, y:yy})
        xx_test,yy_test=get_batch(x_test, y_test, batch_size=64)
#        loss_value, acc = sess.run([loss, accuracy], feed_dict={x:mnist.test.images, y:mnist.test.labels})
        loss_value, acc = sess.run([loss, accuracy], feed_dict={x:xx_test, y:yy_test})
        # 始终打印最好的准确率信息
        if acc > best:
            best = acc
            last_batch  = total_batch
            print('epoch:%d, loss:%f, acc:%f' % (epoch, loss_value, acc))
        if total_batch - last_batch > 5: # 训练早停条件
            print('when epoch-%d early stop train'%epoch)
            break
        total_batch += 1