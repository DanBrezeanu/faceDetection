from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()

ap.add_argument('-p', '--prototxt', required=True, help='Path to caffe deploy prototxt file')
ap.add_argument('-v', '--video', required=True, help='Path to video file')
ap.add_argument('-m', '--model', required=True, help='path to Caffe pre-trained model')
ap.add_argument('-c', '--confidence', type=float, default=0.5,
                help='minimum probability to filter weak detections')

args = vars(ap.parse_args())

# load model from disk

print('[INFO] loading model...')

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print('[INFO] starting video stream...')

fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

vs = FPS().start()

while fvs.more():
    frame = fvs.read()

    frame = imutils.resize(frame, width=450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])

    (h, w) = frame.shape[:2]

    # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.7, 177.0, 123.0), True)

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < args["confidence"]:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

    vs.update()

vs.stop()
fvs.release()
cv2.destroyAllWindows()