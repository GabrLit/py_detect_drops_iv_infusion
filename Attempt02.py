import cv2
import cv2 as cv
import imutils
import argparse
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="video source")
ap.add_argument('-o', "--orig", action=argparse.BooleanOptionalAction)
ap.add_argument('-g', "--gray", action=argparse.BooleanOptionalAction)
ap.add_argument('-b', "--bin", action=argparse.BooleanOptionalAction)
ap.add_argument("-w", "--wait", required=False, help="waitTimeBetweenFrames")
args = vars(ap.parse_args())

firstFrame = None
cap = cv.VideoCapture(args["source"])
fgbg = cv2.createBackgroundSubtractorMOG2()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize frame and convert to grayscale
    frame = imutils.resize(frame, width=700)
    frameGray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameGray = cv.GaussianBlur(frameGray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = frameGray

    # MOG Gaussian
    fgmask = fgbg.apply(frameGray)

    # Dilate images to have full drops
    dilation = cv.dilate(fgmask, np.ones((7, 7), np.uint8), iterations=2)

    # Detect contours
    contours, hierarchy = cv2.findContours(image=dilation, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # Draw bounding boxes
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print(x, y, w, h)

    # cv2.drawContours(image=frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
    #                  lineType=cv2.LINE_AA)

    cv2.imshow("boundingBoxes", dilation)

    if args["orig"]:
        cv.imshow("original", frame)
    if args["gray"]:
        cv.imshow("gray", frameGray)
    # if args["bin"]:
    #     cv.imshow("binary", frameThresh)

    cv.waitKey(int(args["wait"]))
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
