import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options = {
    'model': 'cfg/tiny-yolo-voc.cfg',
    'load': 'bin/tiny-yolo-voc.weights',
    'threshold': 0.2,
    'gpu': 0.78
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

centroid = []

def direction_vertical(w2, h2,c_box2 ):

    if((h2+200) < c_box2[1]):
        print("Down")

    elif((h2-200) > c_box2[1]):
        print("Up")

    else:
        print("vertical still")

    print(" ")


def direction_horizontal(w,h,c_box):



    if ((w+200) < c_box[0]):
        print("Left")

    elif((w-200) > c_box[0]):
        print("Right")

    else:
        print("horizontal still")

    print(" ")




def centre_of_box(top_left, bottom_right):
    x = top_left[0] + (bottom_right[0] - top_left[0])/2
    y = top_left[1] + (bottom_right[1] - top_left[1])/2

    x = int(x)
    y = int(y)

    cord = [x, y]

    return cord



while True:
    stime = time.time()
    ret, frame = capture.read()

    if ret:
        results = tfnet.return_predict(frame)
        height, width = frame.shape[0:2]
        height = int(height / 2)
        width = int(width / 2)

        print(results)
        print(" ")

        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            #print(tl)
            #print(br)
            centroid = centre_of_box(tl, br)
            #print(centroid[0])
            frame = cv2.rectangle(frame, tl, br, (255,255,51), 5)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            frame = cv2.circle(frame, (centroid[0], centroid[1]), 20, (255,255,51), -1)
            direction_horizontal(width, height, centroid)
            direction_vertical(width, height, centroid)
        #print(frame.shape[0:2])
        #print(tl)
        #print(width)
        cv2.circle(frame, (width, height), 20, (0, 255, 0), -1)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()