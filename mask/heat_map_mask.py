import cv2 as cv
import numpy as np
import os

# Mappa beállítása
video_folder = 'videos/Rat2/'
video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.dav')]

if not video_files:
    print("Nincs feldolgozható videó a mappában.")
    exit(0)

# Háttérlevonó inicializálása
bg_subtractor = cv.createBackgroundSubtractorKNN(history=800, dist2Threshold=800, detectShadows=False)

# Hőtérkép inicializálása
motion_heatmap = None
frame_count = 0

# Videók feldolgozása egyesével
for video_path in video_files:
    print(f'Feldolgozás: {video_path}')
    capture = cv.VideoCapture(video_path)

    if not capture.isOpened():
        print(f'Nem sikerült megnyitni: {video_path}')
        continue

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame_count += 1
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Mozgásdetektálás háttérkivonással
        fg_mask = bg_subtractor.apply(gray_frame)

        # A maszk invertálása (a patkány legyen fekete, a háttér fehér)
        fg_mask = cv.bitwise_not(fg_mask)

        # Hőtérkép frissítése
        if motion_heatmap is None:
            motion_heatmap = np.zeros_like(fg_mask, dtype=np.float32)

        motion_heatmap += fg_mask.astype(np.float32)

    capture.release()  # Videó lezárása

# **2. lépés: A hőtérkép normalizálása (0-255 skálára)**
motion_heatmap = cv.normalize(motion_heatmap, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

# **Szürkeárnyalatos maszk mentése**
cv.imwrite("rat_movement_mask.png", motion_heatmap)
print("Szürkeárnyalatos mozgásmaszk mentve: rat_movement_mask.png")

cv.destroyAllWindows()
