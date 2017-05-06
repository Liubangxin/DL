# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print('loaded!')

    # 定义输入X
    X = tf.placeholder(tf.float32, shape=[None, 784])
    # 参数变量的定义
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # 定义模型
    y = tf.nn.softmax(tf.matmul(X, w) + b)

    # 训练模型
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    # 定义损失函数，交叉熵
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    # 梯度下降
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # 创建一个操作，初始化所有变量
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={X: batch_xs, y_: batch_ys})

    # 评估模型
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, y_: mnist.test.labels}))