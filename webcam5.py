import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
from pyimagesearch.centroidtracker import CentroidTracker

import JETSON.GPIO as GPIO

options = {
    'model': 'cfg/tiny-yolo-voc.cfg',
    'load': 'bin/tiny-yolo-voc.weights',
    'threshold': 0.6,
    'gpu': 0.78
}

ha_pin = 5
hd_pin = 6
va_pin = 13
vd_pin = 19
mid_pin = 26

GPIO.setmode(GPIO.BOARD)
GPIO.setup(ha_pin, GPIO.OUT)
GPIO.setup(hd_pin, GPIO.OUT)
GPIO.setup(va_pin, GPIO.OUT)
GPIO.setup(vd_pin, GPIO.OUT)
GPIO.setup(mid_pin, GPIO.OUT)

stat = 0
time2 = 0

ct = CentroidTracker()

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

centroid = []
new_IDs = []
new2_IDs = []
new2_centres = []
id_centres = []

ID_time = 0

def direction_vertical(w,h,hor, ver):

    if((h+200) <= ver):
        GPIO.output(va_pin, GPIO.LOW)
        GPIO.output(vd_pin, GPIO.LOW)
        print("Down")
        return 2

    elif((h-200) >= ver):
        GPIO.output(va_pin, GPIO.HIGH)
        GPIO.output(vd_pin, GPIO.HIGH)
        print("Up")
        return 2


    else:
        print("vertical still")
        return 1

    print(" ")


def direction_horizontal(w,h,hor, ver):

    if ((w+200) <= hor):

        GPIO.output(ha_pin, GPIO.LOW)
        GPIO.output(hd_pin, GPIO.LOW)

        print("Left")
        return 2

    elif((w-200) >= hor):
        GPIO.output(ha_pin, GPIO.HIGH)
        GPIO.output(hd_pin, GPIO.HIGH)

        print("Right")
        return 2

    else:
        print("horizontal still")
        return 1

    print(" ")


def go_to_middle(id, centre, w, h):

    h_stat = 0
    v_stat = 0


    for i in range(len(id)):
        #stat = check_status(id[i], centre[count], w, h)
        x = centre[i][0]
        y = centre[i][1]

        h_stat = direction_horizontal(w, h, x, y)
        v_stat = direction_vertical(w, h, x, y)

        if(h_stat == 2 and v_stat == 2):
            print('Same')
            return 'm'

        else:
            print('Different')
            return 'nm'

        print(id[i])
        print(centre[i])


while True:
    stime = time.time()
    ret, frame = capture.read()

    if ret:
        results = tfnet.return_predict(frame)
        height, width = frame.shape[0:2]
        height = int(height / 2)
        width = int(width / 2)

        rects = []

        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            if (label == "person"):
                centroid = centre_of_box(tl, br)
                box = np.array([tl[0],tl[1], br[0], br[1]])
                rects.append(box.astype("int"))
                frame = cv2.rectangle(frame, tl, br, (255,255,51), 5)
                frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                #frame = cv2.circle(frame, (centroid[0], centroid[1]), 20, (255,255,51), -1)
                #direction_horizontal(width, height, centroid)
                #direction_vertical(width, height, centroid)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            full_list = objects.keys()
            centroid_list = objects.values()

            for i in full_list:
                new2_IDs.append(i)


            for c in centroid_list:
                new2_centres.append(c)


            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 10, (0, 255, 0), -1)

            if objectID not in new_IDs:
                new_IDs.append(objectID)
                id_centres.append(centroid)

            stat = go_to_middle(new2_IDs, new2_centres, width, height)

            if(stat == "m"):

                GPIO.output(mid_pin, GPIO.HIGH)

                if((time.time() - stime) > 2):
                    GPIO.output(mid_pin, GPIO.LOW)

            elif(stat == "nm"):
                GPIO.output(mid_pin, GPIO.LOW)

            diff = time.time() - stime
            ID_time += diff

            if(ID_time >= 2):
                #print(objectID)
                #print('{:.1f}'.format(1 / (time.time() - stime)))
                stime = time.time()
                ID_time = 0

            new2_IDs = []
            new2_centres =[]
        #print(frame.shape[0:2])
        #print(tl)
        #print(width)
        cv2.circle(frame, (width, height), 10, (0, 255, 0), -1)
        cv2.imshow('frame', frame)
        #print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()