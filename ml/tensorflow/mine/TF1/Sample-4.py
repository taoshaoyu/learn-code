# TF1 + mnist
# Can not run
# https://zhuanlan.zhihu.com/p/30572900
# FIXBUG: what is input_tensor

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()

# MNIST数据集相关的常数。
INPUT_NODE = 784      # 输入层的节点数。对于MNIST数据集，这个就等于图片的像素。   
OUTPUT_NODE = 10     # 输出层的节点数。这个等于类别的数目。因为在MNIST数据集中
                         # 需要区分的是0~9这10个数字，所以这里输出层的节点数为10。

# 配置神经网络的参数。
LAYER1_NODE = 500   # 隐藏层节点数。这里使用只有一个隐藏层的网络结构作为样例。
                        # 这个隐藏层有500个节点。
BATCH_SIZE = 100    # 一个训练batch中的训练数据个数。数字越小时，训练过程越接近
                        # 随机梯度下降；数字越大时，训练越接近梯度下降。
LEARNING_RATE = 0.01           # 学习率。
TRAINING_STEPS = 10000              # 训练轮数。

def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    y_ret_data=[]
    for i in y_data[idxs]:
        tmp=np.zeros(10)
        tmp[i]=1
        y_ret_data.append(tmp)
    return x_data[idxs,:,:].reshape(batch_size,28*28), np.array(y_ret_data)

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    
    # 定义神经网络参数。
    weights1 = tf.Variable(
    tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    bias1 = tf.Variable(tf.constant(0.0, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(
    tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.0, shape=[OUTPUT_NODE]))

# 计算在当前参数下神经网络前向传播的结果。
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + bias1)
    y = tf.matmul(layer1, weights2) + bias2

# 定义存储训练轮数的变量。 
    global_step = tf.Variable(0, trainable=False)
    
# 计算交叉熵作为刻画预测值和真实值之间差距的损失函数。
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=y_, logits=y)
    loss = tf.reduce_mean(cross_entropy)
           
    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数。注意这里损失
    # 函数包含了交叉熵损失和L2正则化损失。
    train_op=tf.train.GradientDescentOptimizer(LEARNING_RATE)\
                 .minimize(loss, global_step=global_step)

    # 检验神经网络的正确率。
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
    # 初始化会话并开始训练过程。
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
    # 准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的
    # 条件和评判训练的效果。
        validate_feed = {x: mnist.validation.images, 
                             y_: mnist.validation.labels}

    # 准备测试数据。在真实的应用中，这部分数据在训练时是不可见的，这个数据只是作为  
    # 模型优劣的最后评价标准。
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}     

        # 迭代地训练神经网络。
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果。
            if i % 1000 == 0:
                    validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %d training step(s), validation accuracy "
                         "using average model is %g " % (i, validate_acc))
            
            # 产生这一轮使用的一个batch的训练数据，并运行训练过程。
            xs, ys = get_batch(x_train, y_train, batch_size=BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率。
        test_acc = sess.run(accuracy, feed_dict=test_feed)
    print("After %d training step(s), test accuracy using average "
           "model is %g" % (TRAINING_STEPS, test_acc))

def main(argv=None): 
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据。
    train(mnist)

if __name__ == '__main__':
    main()