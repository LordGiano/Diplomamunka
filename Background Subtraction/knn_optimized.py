from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import os
from skopt import gp_minimize
from skopt.space import Integer, Real

# Argumentum parser
parser = argparse.ArgumentParser(description='Automatikus paraméterhangolás KNN háttérkivonással.')
parser.add_argument('--input', type=str, help='Videó fájl elérési útja.',
                    default='../videos/Rat1/08.07.02-08.09.03[M][0@0][0].dav')
parser.add_argument('--detectShadows', type=bool, help='Árnyékdetektálás beállítása.', default=False)
args = parser.parse_args()


# Képernyőkép készítési funkció
def save_screenshot(window_name, frame, filtered_frame, frame_index):
    video_path = args.input
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]

    parts = video_name.split('[')[0]
    rat_folder = os.path.basename(os.path.dirname(video_path))

    directory = os.path.join(window_name, rat_folder)
    filename = os.path.join(directory, f"{parts}_FRAME_{frame_index}_combined.png")

    if len(frame.shape) == 3 and len(filtered_frame.shape) == 2:
        filtered_frame = cv.cvtColor(filtered_frame, cv.COLOR_GRAY2BGR)

    combined_frame = cv.hconcat([frame, filtered_frame])

    os.makedirs(directory, exist_ok=True)

    success = cv.imwrite(filename, combined_frame)
    if success:
        print(f"Képernyőkép elmentve: {filename}")
    else:
        print(f"Hiba: Nem sikerült a képernyőkép mentése: {filename}")


# Új metrika: kis zajos foltok szűrése
def evaluate_knn(params):
    history, dist2Threshold = params

    # Újra megnyitjuk a videót minden próbánál
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
    if not capture.isOpened():
        return float("inf")

    backSub = cv.createBackgroundSubtractorKNN(history=int(history), dist2Threshold=dist2Threshold,
                                               detectShadows=args.detectShadows)

    total_noise = 0
    total_valid_area = 0
    frame_count = 0

    while frame_count < 200:
        ret, frame = capture.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)

        # Kis zajos foltok számolása kontúrdetektálással
        contours, _ = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)

        # Értékelés: kevesebb kis zajos folt, nagyobb mozgó objektumok
        total_noise += num_contours

        # Nagyobb objektumok számolása
        valid_contours = [c for c in contours if cv.contourArea(c) > 500]  # Csak nagyobb objektumokat számoljuk
        total_valid_area += len(valid_contours)

        frame_count += 1

    capture.release()

    # Cél: minimalizálni a zajt, maximalizálni a nagy objektumokat
    return total_noise - 2 * total_valid_area


# Optimalizációs keresési tér (szűkebb intervallumok)
space = [Integer(200, 1000, name="history"), Real(300.0, 2500.0, name="dist2Threshold")]

# Optimalizáció végrehajtása
res = gp_minimize(evaluate_knn, space, n_calls=25, random_state=42)
best_history, best_dist2Threshold = res.x

print(f"Optimális paraméterek: history={best_history}, dist2Threshold={best_dist2Threshold}")

# Optimalizált háttérkivonó
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
backSub = cv.createBackgroundSubtractorKNN(history=int(best_history), dist2Threshold=best_dist2Threshold,
                                           detectShadows=args.detectShadows)

paused = False
last_frame = None
last_fgMask = None
last_filtered_fgMask = None

while True:
    if not paused:
        ret, frame = capture.read()
        if frame is None:
            break

        frame_index = int(capture.get(cv.CAP_PROP_POS_FRAMES))
        fgMask = backSub.apply(frame)
        filtered_fgMask = cv.medianBlur(fgMask, 5)

        last_frame = frame.copy()
        last_fgMask = fgMask.copy()
        last_filtered_fgMask = filtered_fgMask.copy()

    if last_frame is not None:
        cv.imshow('Original Video', last_frame)
    if last_fgMask is not None:
        cv.imshow('Foreground Mask', last_fgMask)
    if last_filtered_fgMask is not None:
        cv.imshow('Processed Video', last_filtered_fgMask)

    keyboard = cv.waitKey(30) & 0xFF
    if keyboard == ord(' '):
        paused = not paused
    elif keyboard == ord('x') or keyboard == ord('X'):
        if last_filtered_fgMask is not None:
            save_screenshot("KNN", last_frame, last_filtered_fgMask, frame_index)
    elif keyboard == 27 or keyboard == ord('q') or keyboard == ord('Q'):
        break

capture.release()
cv.destroyAllWindows()
