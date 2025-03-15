import cv2 as cv
import numpy as np

# Videó beolvasása
capture = cv.VideoCapture('../videos/Rat2/08.06.59-08.08.52[M][0@0][0].dav')
if not capture.isOpened():
    print('Unable to open video')
    exit(0)

# Az első képkocka beolvasása
ret, first_frame = capture.read()
if not ret:
    print('Unable to read the first frame')
    exit(0)

# Szürkeárnyalatos első kép
prev_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
height, width = prev_frame.shape

# Alap maszk létrehozása (fekete kép)
grid_mask = np.zeros((height, width), dtype=np.uint8)

# Képkockák számlálása
frame_count = 0

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Képkockák közötti különbség
    diff_frame = cv.absdiff(prev_frame, gray_frame)

    # Zajszűrés - Gauss és medián szűrés
    blurred_diff = cv.GaussianBlur(diff_frame, (5, 5), 0)
    median_diff = cv.medianBlur(blurred_diff, 5)

    # Képkockák küszöbölése
    _, thresh_diff = cv.threshold(median_diff, 20, 255, cv.THRESH_BINARY)

    # A maszk építése az állandó háttérrészekkel
    grid_mask = cv.bitwise_or(grid_mask, thresh_diff)

    # Előző kép frissítése
    prev_frame = gray_frame
    frame_count += 1

# Morfológiai műveletek
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
grid_mask = cv.morphologyEx(grid_mask, cv.MORPH_CLOSE, kernel)
grid_mask = cv.morphologyEx(grid_mask, cv.MORPH_OPEN, kernel)

# Maszk mentése és megjelenítése
cv.imwrite('grid_mask_output.png', grid_mask)
cv.imshow('Final Grid Mask', grid_mask)
cv.waitKey(0)

capture.release()
cv.destroyAllWindows()
