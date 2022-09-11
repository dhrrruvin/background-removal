import os
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from data import load_data, tf_dataset
from model import build_unet


np.random.seed(42)
tf.random.set_seed(42)

train_path = "dataset/train"
test_path = "dataset/test"
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(train_path, test_path)


shape = (256, 256, 3)
num_classes = 3
lr = 1e-4
batch_size = 8
epochs = 10

model = build_unet(shape, num_classes)
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

train_dataset = tf_dataset(train_x, train_y, batch_size)
valid_dataset = tf_dataset(valid_x, valid_y, batch_size)

train_steps = len(train_x)//batch_size
valid_steps = len(valid_x)//batch_size

callbacks = [
    ModelCheckpoint("model.h5", verbose=1, save_best_only=True)
]

model.fit(train_dataset, steps_per_epoch=train_steps, validation_data=valid_dataset, validation_steps=valid_steps, epochs=epochs, callbacks=callbacks)