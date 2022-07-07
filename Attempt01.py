import cv2 as cv
import imutils
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="video source")
ap.add_argument('-o', "--orig", action=argparse.BooleanOptionalAction)
ap.add_argument('-d', "--diff", action=argparse.BooleanOptionalAction)
ap.add_argument('-b', "--bin", action=argparse.BooleanOptionalAction)
ap.add_argument("-w", "--wait", required=False, help="waitTimeBetweenFrames")
args = vars(ap.parse_args())

cap = cv.VideoCapture(args["source"])
ret, frameRaw = cap.read()
# resize frame
frameResized = imutils.resize(frameRaw, 700)
frame = cv.cvtColor(frameResized, cv.COLOR_BGR2GRAY)

while cap.isOpened():
    prevFrame = frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # resize frame
    frameRaw = imutils.resize(frame, 700)
    frameResized = cv.cvtColor(frameRaw, cv.COLOR_BGR2GRAY)
    frame = cv.GaussianBlur(frameResized, (3, 3), 0)

    difImage = cv.absdiff(prevFrame, frame)
    blur = cv.medianBlur(difImage, 15)
    ret, thresh = cv.threshold(blur, 2, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    if args["bin"]:
        cv.imshow("binary", thresh)
    if args["diff"]:
        cv.imshow("difference", difImage)
    if args["orig"]:
        cv.imshow("original", frameRaw)
    cv.waitKey(int(args["wait"]))

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
