import os
import cv2 as cv
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from data import load_data, tf_dataset
from model import build_unet

num_classes = 3

np.random.seed(42)
tf.random.set_seed(42)

train_path = "dataset/train"
test_path = "dataset/test"
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(train_path, test_path)

model = tf.keras.models.load_model("model.h5")

for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
    name = x.split("/")[-1]

    x = cv.imread(x, cv.IMREAD_COLOR)
    x = cv.resize(x, [256, 256])
    x = x / 255.0
    x = x.astype(np.float32)

    y = cv.imread(y, cv.IMREAD_COLOR)
    y = cv.resize(y, [256, 256])
    y = y-1
    y = np.expand_dims(y, axis=-1)
    y = y*(255/num_classes)
    y = y.astype(np.int32)
    y = np.concatenate([y, y, y], axis=2)

    p = model.predict(np.expand_dims(x, axis=0))[0]
    p = np.argmax(p, axis=-1)
    p = np.expand_dims(p, axis=-1)
    p = p*(255/num_classes)
    p = p.astype(np.int32)
    p = np.concatenate([p, p, p], axis=2)

    x = x*255.0
    x = x.astype(np.int32)

    final_image = np.concatenate([x, y, p], axis=1)
    cv.imwrite(f"dataset/output/{name}", final_image)