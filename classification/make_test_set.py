import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.

import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

TEST_DIR = 'E:\Project\Ai_Club\dataSets\\train'
TEST_DIR = 'E:\Project\Ai_Club\dataSets\\test'
IMG_SIZE = 50
LR = 1e-3


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat':
        return [1, 0]
    #                             [no cat, very doggo]
    elif word_label == 'dog':
        return [0, 1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img)
        path = os.path.join(TEST_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('./data/train_data.npy', training_data)
    return training_data

def create_test_data():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img)
        path = os.path.join(TEST_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        test_data.append([np.array(img), np.array(label)])
    shuffle(test_data)
    np.save('./data/test2_data.npy', test_data)
    return test_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('./data/test_data.npy', testing_data)
    return testing_data


# test_data = process_test_data()
# train_data = create_train_data()
test_data2 = create_test_data()
