"""
this program is used to plot accuracy and loss graph to learn more about training
and computes IoU(intersection over union) to find the accuracy of prediction
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_graph(history, save=False):
    print(history.history.keys())

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.savefig("accuracy.jpg")
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig("loss.jpg")
    plt.show()

def compute_iou(output, mask):

    SMOOTH = 1e-6

    output = output.squeeze(1)

    intersection = (output & mask).sum((1, 2))
    union = (output | mask).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    threshold = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return threshold
