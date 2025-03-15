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

# Videók feldolgozása egyesével
for video_path in video_files:
    print(f'Feldolgozás: {video_path}')
    capture = cv.VideoCapture(video_path)

    if not capture.isOpened():
        print(f'Nem sikerült megnyitni: {video_path}')
        continue

    # Maszk inicializálása az adott videóhoz
    video_mask = None

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Rácsdetektálás
        blurred = cv.GaussianBlur(gray_frame, (5, 5), 0)
        edges = cv.Canny(blurred, 20, 80)  # Alacsonyabb küszöbérték, érzékenyebb detektálás

        if video_mask is None:
            video_mask = np.zeros_like(edges, dtype=np.uint8)

        video_mask = cv.bitwise_or(video_mask, edges)  # Az adott videó maszkját folyamatosan építjük

    capture.release()  # Videó lezárása

    # Morfológiai műveletek a videó maszkon
    if video_mask is not None:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
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
            final_mask = cv.bitwise_and(final_mask, video_mask)  # Csak a közös részek maradjanak meg

# A végső maszk mentése
if final_mask is not None:
    cv.imwrite("grid_mask_calculated.png", final_mask)
    print("Végső rácsmaszk mentve: grid_mask_calculated.png")

cv.destroyAllWindows()
