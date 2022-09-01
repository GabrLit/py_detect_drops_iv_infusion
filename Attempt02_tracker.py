import cv2
import cv2 as cv
import imutils
import argparse
import numpy as np
from tracker import *
import os
import time
from pprint import pprint


class BoxGroup:
    def __init__(self, id):
        self.id = id
        self.posAndFrames = []

    def append(self, arr):
        self.posAndFrames.append(arr)


MINIMAL_Y_POS_DIFF = 35  # Minimal distance in px that drop has to travel
DROP_VOLUME = 0.05  # Drop volume 0.05ml
FRAMERATE = 30  # FPS
APPROXIMATE = 10  # How many records to approximate

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="video source")
ap.add_argument('-o', "--orig", action=argparse.BooleanOptionalAction)
ap.add_argument('-g', "--gray", action=argparse.BooleanOptionalAction)
ap.add_argument('-b', "--bin", action=argparse.BooleanOptionalAction)
ap.add_argument("-w", "--wait", required=False, help="waitTimeBetweenFrames")
ap.add_argument("-v", "--volume", required=True, help="volume of medicine")
args = vars(ap.parse_args())

# Create tracker object
tracker = EuclideanDistTracker()

boxGroups = []

firstFrame = None
frameId = 0
cap = cv.VideoCapture(args["source"])
fgbg = cv2.createBackgroundSubtractorMOG2()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frameId = frameId + 1

    if firstFrame is None:
        firstFrame = imutils.resize(frame, width=700)

    # Resize frame and convert to grayscale
    print(frame.shape)
    frame = imutils.resize(frame, width=700)
    cv2.imshow('orig', frame)

    frameBox = frame.copy()
    frameGray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameGrayBlur = cv.GaussianBlur(frameGray, (25, 25), 0)

    # MOG Gaussian
    fgmask = fgbg.apply(frameGrayBlur)

    # Dilate images to have full drops
    dilation = cv.dilate(fgmask, np.ones((9, 9), np.uint8), iterations=3)

    # Detect contours
    contours, hierarchy = cv2.findContours(image=dilation, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # Detections
    detections = []
    lastWrittenFrame = None

    # Draw bounding boxes
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        detections.append([x, y, w, h])

    # Object tracking
    boxesIds = tracker.update(detections)

    # Prepare box groups
    if len(detections) > 0:
        # Iterate over array
        for record in boxesIds:
            recordId = record.pop(-1)

            # Check if id already exists
            idExists = False
            for boxGroup in boxGroups:
                if boxGroup.id == recordId:
                    idExists = True
                    # If this frame was already written for the specific id then break
                    if lastWrittenFrame == frameId:
                        break
                    boxGroup.posAndFrames.append([record, frameId])

            if not idExists:
                newBoxGroup = BoxGroup(recordId)
                newBoxGroup.append([record, frameId])
                boxGroups.append(newBoxGroup)

            lastWrittenFrame = frameId

    # cv2.imshow("boundingBoxes", dilation)
    # if args["orig"]:
    #     cv.imshow("original", frame)
    # # if args["gray"]:
    # #     cv.imshow("gray", frameGray)
    # #
    # # os.system('clear')
    # # for boxGroup in boxGroups:
    # #     print(vars(boxGroup))
    # #

    cv2.imshow("grayscale", frameGray)
    cv2.imshow('blur', frameGrayBlur)
    cv2.imshow('backgroundsub', fgmask)
    cv2.imshow('morphology', dilation)
    cv2.drawContours(frameBox, contours, -1, (0, 255, 0), 2)
    cv2.imshow("detection", frame)

    cv.waitKey()
    if cv.waitKey(int(args["wait"])) == ord('q'):
         break
    os.system('clear')
    print("stage 1/2 Frame: ", frameId)
    print("\nseconds: ", int(frameId / 30))


# Filter out box groups that have only one record of posAndFrames
def filterOutSingleRecord(bGroup):
    if len(bGroup.posAndFrames) > 1:
        return True
    return False


def filterOutSmallYDiff(bGroup):
    boxCenter = bGroup.posAndFrames[0][0][1] + int(bGroup.posAndFrames[0][0][3])
    MIN = MAX = boxCenter
    for recrd in bGroup.posAndFrames:
        if recrd[0][1] + int(recrd[0][3]) < MIN:
            MIN = recrd[0][1] + int(recrd[0][3])
        if recrd[0][1] + int(recrd[0][3] / 2) > MAX:
            MAX = recrd[0][1] + int(recrd[0][3])

    return MAX - MIN > MINIMAL_Y_POS_DIFF


bGroups = filter(filterOutSingleRecord, boxGroups)
bGroups = filter(filterOutSmallYDiff, bGroups)
# Reformat Ids
bGroups = [{'id': index + 1, 'posAndFrames': bGroup.posAndFrames} for index, bGroup in enumerate(bGroups)]

cap.release()

# Display result
cap = cv.VideoCapture(args["source"])
dir = args["source"].split('/').pop()
writer = cv2.VideoWriter("results/" + dir.split('.')[0] + '.avi', cv2.VideoWriter_fourcc(*"MJPG"), 30,
                         (firstFrame.shape[1], firstFrame.shape[0]))

averageTab = []
lastDropId = -1
lastDropInitialFrame = -1

frameId = 0
totalCounter = 0
tempoCounter = 0
volumeLeft = float(args['volume'])
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frameId = frameId + 1

    frame = imutils.resize(frame, width=700)
    for boxGroup in bGroups:
        for posAndFrame in boxGroup['posAndFrames']:
            if posAndFrame[-1] == frameId:
                [x, y, w, h] = posAndFrame[0]
                frame = cv2.circle(frame, (x + int(w / 2), y + int(h / 2)), radius=6, color=(0, 255, 0), thickness=-1)
                totalCounter = boxGroup['id']

                # to calculate tempo:
                if lastDropId != boxGroup['id']:
                    if lastDropInitialFrame != -1:
                        tempoCounter = FRAMERATE / (frameId - lastDropInitialFrame)
                        volumeLeft = volumeLeft - DROP_VOLUME

                    lastDropId = boxGroup['id']
                    lastDropInitialFrame = frameId
                break

    frame = cv2.putText(frame, "Licznik kropel: " + str(totalCounter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 4, cv2.LINE_AA)

    frame = cv2.putText(frame, "Licznik kropel: " + str(totalCounter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 0, 0), 1, cv2.LINE_AA)

    # average the results
    if len(averageTab) < APPROXIMATE:
        print("This is called only once")
        for i in range(APPROXIMATE):
            averageTab.append(tempoCounter)

    averageTab.append(tempoCounter)
    averageTab.pop(0)

    accumulator = 0.0
    average = 0.0
    for i in averageTab:
        accumulator = accumulator + i

    average = accumulator / APPROXIMATE

    frame = cv2.putText(frame, "Krople na sekunde: " + "{:.2f}".format(average), (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255), 3, cv2.LINE_AA)

    frame = cv2.putText(frame, "Krople na sekunde: " + "{:.2f}".format(average), (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 0, 0), 1, cv2.LINE_AA)

    approx = 0
    if average * DROP_VOLUME != 0:
        approx = float(volumeLeft) / (average * DROP_VOLUME)

    frame = cv2.putText(frame,
                            "{:.2f}".format(volumeLeft) + "ml przyblizony czas zakonczenia: " + time.strftime(
                                '%H:%M:%S',
                                time.gmtime(
                                    approx)),
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (255, 255, 255), 3, cv2.LINE_AA)

    frame = cv2.putText(frame,
                        "{:.2f}".format(volumeLeft) + "ml przyblizony czas zakonczenia: " + time.strftime('%H:%M:%S',
                                                                                                       time.gmtime(
                                                                                                           approx)),
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 0, 0), 1, cv2.LINE_AA)

    # Too slow, too fast:
    if average < 2.0 or frameId - lastDropInitialFrame > FRAMERATE:
        frame = cv2.putText(frame, "Za wolno!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
        frame = cv2.putText(frame, "Za wolno!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1, cv2.LINE_AA)

    if average > 3.0:
        frame = cv2.putText(frame, "Za szybko!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255, 255), 3, cv2.LINE_AA)
        frame = cv2.putText(frame, "Za szybko!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1, cv2.LINE_AA)

    # cv2.imshow("result", frame)
    # cv.waitKey()
    # if cv.waitKey(1) == ord('q'):
    #     break
    os.system('clear')
    print("stage 2/2 Frame: ", frameId)
    print("\nseconds: ", int(frameId / 30))

    writer.write(frame)

cap.release()
writer.release()
cv.waitKey(0)
cv.destroyAllWindows()
