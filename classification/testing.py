# Author: sihan
# Date: 2018-09-14

import tensorflow as tf
from PIL import Image
import numpy as np
import os
import glob

os.environ['TF_CPP+MIN_LOG_LEVEL'] = '3'

image_width = 64
image_height = 64
# 지도 학습을 위한 라벨
image_label = []
# 리사이징 된 이미지
All_image = []


input_data = []


# img = Image.open("./data/a_set\dog.4001.jpg")
# img = img.resize((image_width, image_height))
#
# input_data.append(np.float32(img))

# 테스트용 데이터 읽어 오기
img_path = os.getcwd()
image_file_list = glob.glob(img_path + "\data\\a_set\*.jpg")

# 이미지 리사이징 및 라벨링
for img in image_file_list:
    # 라벨링
    if 'kitten' in img:
        image_label.append(0)
    else:
        image_label.append(1)

    image = Image.open(img)
    image = image.resize((image_width, image_height))
    All_image.append(np.float32(image))
label = np.eye(2)[image_label]

# 예측값


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
    # 실제값
    original = tf.argmax(Y, 1)

    # 일치 하는지 확인
    is_correct = tf.equal(predict, original)
    # 정확도
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    print(sess.run(predict, feed_dict={X: All_image, keep_prob: 1}))
    print(sess.run(original, feed_dict={Y: label}))

    print("ACC >> ", sess.run(accuracy, feed_dict={X: All_image, Y: label, keep_prob: 1}))

    # predict = tf.argmax(model, 1)
    # predictions = sess.run(predict, feed_dict={X: input_data, keep_prob: 1.0})
    # print(predictions[0])
    #
    # if predictions[0] == 0:
    #     print("cat")
    # else:
    #     print("dog")
