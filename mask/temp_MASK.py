import cv2 as cv
import numpy as np
import os

# Mappa beállítása
video_folder = '../videos/Rat2/'
video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.dav')]

if not video_files:
    print("Nincs feldolgozható videó a mappában.")
    exit(0)

final_mask = None  # A végső maszk kezdetben üres

# Háttérlevonó inicializálása
#bg_subtractor = cv.createBackgroundSubtractorKNN(history=800, dist2Threshold=800, detectShadows=False)

# Videók feldolgozása egyesével
for video_path in video_files:
    bg_subtractor = cv.createBackgroundSubtractorKNN(history=800, dist2Threshold=800, detectShadows=False)
    print(f'Feldolgozás: {video_path}')
    capture = cv.VideoCapture(video_path)

    if not capture.isOpened():
        print(f'Nem sikerült megnyitni: {video_path}')
        continue

    # Maszk inicializálása fehér háttérrel
    video_mask = None

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Mozgásdetektálás háttérkivonással
        fg_mask = bg_subtractor.apply(gray_frame)
        #cv.imshow("Maszk", fg_mask)
        # print(fg_mask.sum()/255)

        # A maszk invertálása (a patkány lesz fekete, a háttér fehér)
        fg_mask = cv.bitwise_not(fg_mask)
        # print("bitwise eredmény: " + str(fg_mask.sum()/255))
        #cv.imshow("Maszk bitwise után", fg_mask)

        #ELSŐ KÉPKOCKÁT ELDOBNI - feketéket ne használjam összehasonlításhoz
        # Kezdeti maszk beállítása
        if video_mask is None:
            video_mask = np.full_like(fg_mask, 255, dtype=np.uint8)  # Fehér háttérrel kezdődik
            # print("video mask INIT: " + str(video_mask.sum() / 255))
        else:
            if fg_mask.sum()/255 > 1_000_000:
                # Az adott videó maszkját folyamatosan frissítjük
                video_mask = cv.bitwise_and(video_mask, fg_mask)
                # print("video mask: " + str(video_mask.sum()/255))

    capture.release()  # Videó lezárása

    # Morfológiai műveletek a videó maszkon (simaítás)
    if video_mask is not None:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        video_mask = cv.morphologyEx(video_mask, cv.MORPH_CLOSE, kernel)
        video_mask = cv.morphologyEx(video_mask, cv.MORPH_OPEN, kernel)

        # Az adott videó maszkjának mentése fájlba
        video_mask_filename = f"mask_{os.path.basename(video_path).replace('.dav', '.png')}"
        cv.imwrite(video_mask_filename, video_mask)
        print(f"Maszk mentve: {video_mask_filename}")

        # A végső maszkhoz hozzáadjuk a videó maszkját
        if final_mask is None:
            final_mask = video_mask.copy()
        else:
            final_mask = cv.bitwise_and(final_mask, video_mask)  # Csak a közös mozgásokat tartja meg

# A végső maszk mentése
if final_mask is not None:
    cv.imwrite("rat_movement_mask.png", final_mask)
    print("Végső mozgásmaszk mentve: rat_movement_mask.png")

cv.destroyAllWindows()
