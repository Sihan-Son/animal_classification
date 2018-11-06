# Created by sihan at 2018-11-03

import os

import cv2
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


class AnimalClassification:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # LR = 1e-4
    # IMG_SIZE = 64
    img_path = ""

    def __init__(self, path, LR=1e-4, IMG_SIZE=64):
        tf.reset_default_graph()
        self.img_path = path
        self.LR = LR
        self.IMG_SIZE = IMG_SIZE

    def run_graph(self):
        MODEL_NAME = 'dogsVScats-{}-{}-{}.model'.format(self.LR, self.IMG_SIZE, '2conv-basic')

        convnet = input_data(shape=[None, self.IMG_SIZE, self.IMG_SIZE, 1], name='input')

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
        convnet = regression(convnet, optimizer='adam', learning_rate=self.LR, loss='categorical_crossentropy',
                             name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        model.load(r'./checkPoint_2conv/' + MODEL_NAME)

        orig = self.resize_img()
        data = orig.reshape(self.IMG_SIZE, self.IMG_SIZE, 1)

        model_out = model.predict([data])

        if np.argmax(model_out) == 1:
            str_label = 'Dog'
        else:
            str_label = 'Cat'

        return str_label

    def resize_img(self):
        img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img_np = np.array(img)

        return img_np
