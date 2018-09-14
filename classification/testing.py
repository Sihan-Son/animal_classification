# Author: sihan
# Date: 2018-09-14

import tensorflow as tf
from PIL import Image
import numpy as np
import os

os.environ['TF_CPP+MIN_LOG_LEVEL'] = '3'

image_width = 64
image_height = 64
input_data = []

img = Image.open("./data/a_set\dogdog.jpg")
img = img.resize((image_width, image_height))

input_data.append(np.float32(img))

# 학습 시킬 변수, input data
X = tf.placeholder(tf.float32, [None, image_width, image_height, 3])
Y = tf.placeholder(tf.float32, [None, 2])

# 학습중 랜덤하게 끊어질 확률 설정
keep_prob = tf.placeholder(tf.float32)

# 은닉층 및 가중치&편향 모델 설계
W1 = tf.Variable(tf.random_normal([3, 3, 3, 128], stddev=-0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([3, 3, 128, 64], stddev=-0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W3 = tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=-0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W4 = tf.Variable(tf.random_normal([8 * 8 * 32, 32], stddev=-0.01))
L4 = tf.reshape(L3, [-1, 8 * 8 * 32])
L4 = tf.matmul(L4, W4)
L4 = tf.nn.relu(L4)
L4 = tf.nn.dropout(L4, keep_prob)

# 츨력층
W5 = tf.Variable(tf.random_normal([32, 2], stddev=0.01))
model = tf.matmul(L4, W5)

# cost 함수 정의
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
# optimizer 정의와 learning_rate 설정
optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(cost)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    save_path = "./checkpoint/animal.ckpt"
    saver.restore(sess, save_path)
    predict = tf.argmax(model, 1)
    predictions = sess.run(predict, feed_dict={X: input_data, keep_prob: 1.0})
    print(predictions[0])

    if predictions[0] == 0:
        print("cat")
    else:
        print("dog")
