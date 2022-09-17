"""
this program removes background from a video file or webcam stream
inorder to run this program change parameter of cv.VideoCapture method
this program does not save the output stream till now
"""
from cgi import test
from time import sleep
import cv2 as cv
import numpy as np
import tensorflow as tf

num_classes = 3

model = tf.keras.models.load_model('model.h5')

cap = cv.VideoCapture('test1.mp4')
while cap.isOpened():
    # sleep(60)
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    img = cv.resize(frame, [256, 256])
    img = img / 255.0
    img = img.astype(np.float32)

    pred = model.predict(np.expand_dims(img, axis=0))[0]
    pred = np.argmax(pred, axis=-1)
    pred = np.expand_dims(pred, axis=-1)
    pred = pred*(255/num_classes)
    pred = pred.astype(np.uint8)
    pred = np.concatenate([pred, pred, pred], axis=2)

    img *= 255 # or any coefficient
    img = img.astype(np.uint8)

    final_output = np.bitwise_and(img, pred)

    cv.imshow("Actual", img)
    cv.imshow("prediction", pred)
    cv.imshow("final", final_output)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

# image = "dataset/test/images/6.png"

# img = cv.imread(image, cv.IMREAD_COLOR)
# img = cv.resize(img, [256, 256])
# img = img / 255.0
# img = img.astype(np.float32)

# pred = model.predict(np.expand_dims(img, axis=0))[0]
# pred = np.argmax(pred, axis=-1)
# pred = np.expand_dims(pred, axis=-1)
# pred = pred*(255/num_classes)
# pred = pred.astype(np.uint8)
# pred = np.concatenate([pred, pred, pred], axis=2)

# img *= 255 # or any coefficient
# img = img.astype(np.uint8)

# print(img.dtype)
# print(pred.dtype)
# print(np.max(img))
# print(np.unique(pred))

# final_output = np.bitwise_and(img, pred)

# cv.imshow("Actual", img)
# cv.imshow("prediction", pred)
# cv.imshow("final", final_output)

# cv.waitKey(0)