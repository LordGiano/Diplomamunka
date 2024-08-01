from __future__ import print_function
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Háttér kivonás képeken és videókon az OpenCV dokumentáció alapján.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='videos\\Rat2\\00.21.38-00.26.49[M][0@0][0].dav')
#parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).')  # KNN
args = parser.parse_args()

# Create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

# Capture the input video
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    # Apply median filter
    median_filtered_frame = cv.medianBlur(frame, 5)

    # Update the background model and get the foreground mask
    fgMask = backSub.apply(median_filtered_frame)

    # Extract the foreground in color
    fgColor = cv.bitwise_and(frame, frame, mask=fgMask)

    # Get the frame number and write it on the current frame
    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Show the current frame and the color foreground mask
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgColor)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

capture.release()
cv.destroyAllWindows()
