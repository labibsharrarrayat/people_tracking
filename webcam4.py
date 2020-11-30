import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
from pyimagesearch.centroidtracker import CentroidTracker

options = {
    'model': 'cfg/tiny-yolo-voc.cfg',
    'load': 'bin/tiny-yolo-voc.weights',
    'threshold': 0.6,
    'gpu': 0.78
}

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

def check_status(j, point, w1, h1):

    print(j)
    print(point[0])
    print(point[1])
    print(w1)
    print(h1)
    #print(point[0])
    if((w1+300)>=point[0] or (w1-300)<= point[0]):
        return 1

    else:
        return 2
        #print('Remains Same')
        #return 'Remain'




def go_to_middle(id, centre, w, h):

    count = 0

    for i in range(len(id)):
        #stat = check_status(id[i], centre[count], w, h)
        x = centre[i][0]
        y = centre[i][1]
        #print(id[i])

        if(x > (w+100) or x <= (w-100)):
            print('Same')

        else:
            print('Different')

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

                #print(tl)
                #print(br)
                centroid = centre_of_box(tl, br)
                #print(centroid[0])
                box = np.array([tl[0],tl[1], br[0], br[1]])
                rects.append(box.astype("int"))
                #print(rects)
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
            #print(new2_IDs)

            for c in centroid_list:
                new2_centres.append(c)
            #print(new2_centres)


            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 10, (0, 255, 0), -1)

            if objectID not in new_IDs:
                new_IDs.append(objectID)
                id_centres.append(centroid)

            go_to_middle(new2_IDs, new2_centres, width, height)
            #go_to_middle(new_IDs, id_centres)

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