from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Háttér kivonás képeken és videókon az OpenCV dokumentáció alapján. '
                                             'Színes megjelenítés. '
                                             'Median filter és inpainting alkalmazása.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='videos\\Rat2\\00.21.38-00.26.49[M][0@0][0].dav')
# parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).')  # KNN vagy MOG2
args = parser.parse_args()

# Háttér kivonó objektum létrehozása
# if args.algo == 'MOG2':
#     backSub = cv.createBackgroundSubtractorMOG2()
# else:
#     backSub = cv.createBackgroundSubtractorKNN()

# Videó vagy kép sorozat megnyitása
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    # Median szűrő alkalmazása
    median_filtered_frame = cv.medianBlur(frame, 7)

    # Készítsünk egy alapértelmezett maszkot az inpaintinghez (opcionálisan más forrásból is jöhet)
    mask = np.zeros(frame.shape[:2], np.uint8)

    # Inpainting alkalmazása a bemeneti képen
    inpainted_frame = cv.inpaint(median_filtered_frame, mask, 5, cv.INPAINT_TELEA)

    # Háttérmodell frissítése és a maszk létrehozása az inpainted frame-en
    # fgMask = backSub.apply(inpainted_frame)

    # Színes előtér kinyerése
    # fgColor = cv.bitwise_and(inpainted_frame, inpainted_frame, mask=fgMask)

    # Keret számának megjelenítése
    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Jelenlegi keret és a színes előtér maszk megjelenítése
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', inpainted_frame)  # Az inpainted_frame-et mutatjuk a FG Mask helyett

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

capture.release()
cv.destroyAllWindows()
