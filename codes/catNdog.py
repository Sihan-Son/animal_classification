# Author: sihan
# Date: 2018-09-10


import tensorflow as tf
from PIL import Image
import glob
import numpy as np
import time
import os

os.environ['TF_CPP+MIN_LOG_LEVEL'] = '3'

# 시간 측정 위한 변수
cur = time.time()
# gpu 맵핑 확인과 메모리 확인을 위한 구문이니 gpu 버전이 아니면 주석 처리하고 사용하세요
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# 리사이징 설정값
image_width = 64
image_height = 64

# 지도 학습을 위한 라벨
image_label = []
# 리사이징 된 이미지
All_image = []

# 이미지 데이터를 불러올 경로
img_path = os.getcwd()
image_file_list = glob.glob(img_path+"\data\\t_set\*.jpg")

# 이미지 리사이징 및 라벨링
for img in image_file_list:
    # 라벨링
    if 'cat' in img:
        image_label.append(0)
    else:
        image_label.append(1)

    # 리사이징
    image = Image.open(img)
    image = image.resize((image_width, image_height))
    All_image.append(np.float32(image))

# one-hot encoding
label = np.eye(2)[image_label]

print("Resizing is end", time.time() - cur, "sec is spend")

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

# 한번에 처리하는 이미지 수
batch_size = 15
# 전체 데이터 반복횟수
training_epochs = 15
# 학습 시간 확인
learning_time = time.time()

# tensorflow 구동을 위해 세션 열기
with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())

    """ 그래프 구조가 바뀌면 주석 풀고 사용해 주세요 """
    # summary = tf.summary.merge_all()
    # writer = tf.summary.FileWriter(r'./boardGraph', sess.graph)

    # 배치 학습
    for epoch in range(training_epochs):
        batch = int(len(All_image) / batch_size)

        for i in range(batch):
            batch_x, batch_y = All_image[i:i + batch], label[i:i + batch]
            _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.7})
        print("epoch- %2d : %.6f" % (epoch + 1, cost_val))
    print(time.time() - learning_time, "sec")

    # 테스트용 데이터 읽어 오기
    image_file_list = glob.glob(img_path+"\data\\a_set\*.jpg")

    # 이미지 리사이징 및 라벨링
    for img in image_file_list:
        # 라벨링
        if 'cat' in img:
            image_label.append(0)
        else:
            image_label.append(1)

        image = Image.open(img)
        image = image.resize((image_width, image_height))
        All_image.append(np.float32(image))
    label = np.eye(2)[image_label]

    # 예측값
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

    print(time.time() - cur, "sec spend")
