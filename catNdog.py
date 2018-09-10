import tensorflow as tf
from PIL import Image
import glob
import numpy as np
import time

cur = time.time()

image_width = 64
image_height = 64
image_label = []
All_image = []

image_file_list = glob.glob("E:\Project\Ai_Club_Project_2018\\animal_classification\\t_set\*.jpg")
for img in image_file_list:
    if 'cat' in img:
        image_label.append(0)
    else:
        image_label.append(1)

    im = Image.open("img")
    im = im.resize((image_width, image_height))
    All_image.append(np.float32(im))
label = np.eye(2)[image_label]

print("Resizing is end")

X = tf.placeholder(tf.float32, [None, image_width, image_height, 3])
Y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)

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
L4 = tf.nn.matmul(L4, W4)
L4 = tf.nn.relu(L4)
L4 = tf.nn.dropout(L4, keep_prob)

W5 = tf.Variable(tf.random_normal([32, 2], stddev=0.01))
model = tf.matmul(L4, W5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


for epoch in range(10):
    _, cost_val = sess.run([optimizer, cost], feed_dict={X: All_image, Y: label, keep_prob: 0.7})
    print("epoch-", epoch + 1, ":", cost_val)
print("enc")

image_file_list = glob.glob("E:\Project\Ai_Club_Project_2018\\animal_classification\\a_set")
for img in image_file_list:
    if 'cat' in img:
        image_label.append(0)
    else:
        image_label.append(1)

    im = Image.open("t_set\cat.4001.jpg")
    im = im.resize((image_width, image_height))
    All_image.append(np.float32(im))
label = np.eye(2)[image_label]

predict = tf.argmax(model, 1)
original = tf.arg_max(Y, 1)
is_correct = tf.equal(predict, original)

print(sess.run(predict, feed_dict={X: All_image, keep_prob: 1}))
print(sess.run(original, feed_dict={Y: image_label}))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("ACC >> ", sess.run(accuracy, feed_dict={X: All_image, Y: image_label, keep_prob: 1}))

print(time.time() - cur, "sec spend")
