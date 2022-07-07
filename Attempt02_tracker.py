import cv2
import cv2 as cv
import imutils
import argparse
import numpy as np
from tracker import *
import os
from pprint import pprint


class BoxGroup:
    def __init__(self, id):
        self.id = id
        self.posAndFrames = []

    def append(self, arr):
        self.posAndFrames.append(arr)


MINIMAL_Y_POS_DIFF = 20  # Minimal distance in px that drop has to travel

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="video source")
ap.add_argument('-o', "--orig", action=argparse.BooleanOptionalAction)
ap.add_argument('-g', "--gray", action=argparse.BooleanOptionalAction)
ap.add_argument('-b', "--bin", action=argparse.BooleanOptionalAction)
ap.add_argument("-w", "--wait", required=False, help="waitTimeBetweenFrames")
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
    frame = imutils.resize(frame, width=700)
    frameGray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameGray = cv.GaussianBlur(frameGray, (25, 25), 0)

    # MOG Gaussian
    fgmask = fgbg.apply(frameGray)

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
    # if args["gray"]:
    #     cv.imshow("gray", frameGray)
    #
    # os.system('clear')
    # for boxGroup in boxGroups:
    #     print(vars(boxGroup))
    #
    # cv2.waitKey()
    # if cv.waitKey(1) == ord('q'):
    #     break


# Filter out box groups that have only one record of posAndFrames
def filterOutSingleRecord(bGroup):
    if len(bGroup.posAndFrames) > 1:
        return True
    return False


def filterOutSmallYDiff(bGroup):
    MIN = MAX = bGroup.posAndFrames[0][0][1]
    for recrd in bGroup.posAndFrames:
        if recrd[0][1] < MIN:
            MIN = recrd[0][1]
        if recrd[0][1] > MAX:
            MAX = recrd[0][1]

    return MAX - MIN > MINIMAL_Y_POS_DIFF


bGroups = filter(filterOutSingleRecord, boxGroups)
bGroups = filter(filterOutSmallYDiff, bGroups)
# Reformat Ids
bGroups = [{'id': index + 1, 'posAndFrames': bGroup.posAndFrames} for index, bGroup in enumerate(bGroups)]

cap.release()

# Display result
cap = cv.VideoCapture(args["source"])
dir = args["source"].split('/').pop()
writer = cv2.VideoWriter("results/" + dir.split('.')[0] + '.avi', cv2.VideoWriter_fourcc(*"MJPG"), 30, (firstFrame.shape[1], firstFrame.shape[0]))
frameId = 0
totalCounter = 0
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
                frame = cv2.putText(frame, str(boxGroup['id']), (x + int(w / 2), y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2, cv2.LINE_AA)
                frame = cv2.circle(frame, (x + int(w / 2), y + int(h / 2)), radius=6, color=(0, 255, 0), thickness=-1)
                totalCounter = boxGroup['id']
                break

    frame = cv2.putText(frame, "Total drop count: " + str(totalCounter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)
    writer.write(frame)
    # cv2.imshow("result", frame)
    # if cv.waitKey(1) == ord('q'):
    #     break

cap.release()
writer.release()
cv.waitKey(0)
cv.destroyAllWindows()
