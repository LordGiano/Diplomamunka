from __future__ import print_function
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Háttér kivonás képeken és videókon az OpenCV dokumentáció alapján. '
                                             'Színes megjelenítés. '
                                             'Median filter alkalmazása.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='videos\\Rat2\\00.21.38-00.26.49[M][0@0][0].dav')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).')  # KNN vagy MOG2
args = parser.parse_args()

# Háttér kivonó objektum létrehozása
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

# Videó megnyitása
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    # Képkocka méretének csökkentése 20%-kal
    width = int(frame.shape[1] * 0.8)
    height = int(frame.shape[0] * 0.8)
    resized_frame = cv.resize(frame, (width, height))

    # Median szűrő alkalmazása
    median_filtered_frame = cv.medianBlur(resized_frame, 7)

    # Háttérmodell frissítése és a maszk létrehozása
    fgMask = backSub.apply(median_filtered_frame)

    # Színes előtér kinyerése
    fgColor = cv.bitwise_and(resized_frame, resized_frame, mask=fgMask)

    # Keret számának megjelenítése
    cv.rectangle(resized_frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(resized_frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Jelenlegi keret és a színes előtér maszk megjelenítése
    cv.imshow('Frame', resized_frame)
    cv.imshow('FG Mask', fgColor)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

capture.release()
cv.destroyAllWindows()
