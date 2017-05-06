# -*- coding:utf-8 -*-

import tensorflow as tf

input = tf.Variable(tf.random_normal([1, 3, 3, 2]))
filter = tf.Variable(tf.random_normal([1, 1, 2, 1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # res = (sess.run(op))
    # print(res.shape)
    i, f, op = sess.run([input, filter, op])
    print(i)
    print(f)
    print(op)