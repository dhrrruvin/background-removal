"""
loading of the images and then preprocessing
"""

import os
import cv2 as cv
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

def process_data(path, image_path, label_path):

    images = [os.path.join(path, image_path, file) for file in os.listdir(os.path.join(path, image_path))]
    masks = [os.path.join(path, label_path, file) for file in os.listdir(os.path.join(path, label_path))]

    return images, masks

def load_data(train_path, test_path):

    train_x, train_y = process_data(train_path, "images", "masks")
    test_x, test_y = process_data(test_path, "images", "masks")

    train_x, valid_x = train_test_split(train_x, test_size=0.2, shuffle=False)
    train_y, valid_y = train_test_split(train_y, test_size=0.2, shuffle=False)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    # return (train_x, train_y), (test_x, test_y)

def read_image(x):
    x = cv.imread(x, cv.IMREAD_COLOR)
    x = cv.resize(x, [256, 256])
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(y):
    y = cv.imread(y, cv.IMREAD_GRAYSCALE)
    y = cv.resize(y, [256, 256])
    y = y-1
    y = y.astype(np.int32)
    return y

def tf_dataset(x, y, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        image = read_image(x)
        mask = read_mask(y)

        return (image, mask)

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, 3, dtype=tf.int32)
    image.set_shape([256, 256, 3])
    mask.set_shape([256, 256, 3])

    return image, mask