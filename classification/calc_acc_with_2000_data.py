# Created by sihan at 2018-11-04

import os

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


IMG_SIZE = 64
LR = 1e-4
MODEL_NAME = 'dogsVScats-{}-{}-{}.model'.format(LR, IMG_SIZE, '2conv-basic')


tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

model.load(r'./checkPoint_2conv/' + MODEL_NAME)
print('model loaded!')

data = np.load(r'./data/test_data_64_2000.npy')

X = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in data]

acc = 0


for i in range(len(Y)):
    model_out = model.predict([X[i]])

    if np.argmax(model_out) == np.argmax(Y[i]):
        acc += 1

print(acc)
print(acc/len(Y))


