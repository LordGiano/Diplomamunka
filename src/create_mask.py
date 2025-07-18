import cv2 as cv
import numpy as np
import os

def process_video(video_path):
    if not os.path.exists(video_path):
        print(f"A fájl nem található: {video_path}")
        return None

    bg_subtractor = cv.createBackgroundSubtractorKNN(history=800, dist2Threshold=800, detectShadows=False)
    print(f'Feldolgozás: {video_path}')
    capture = cv.VideoCapture(video_path)

    if not capture.isOpened():
        print(f'Nem sikerült megnyitni: {video_path}')
        return None

    video_mask = None  # Maszk inicializálása fehér háttérrel

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Mozgásdetektálás háttérkivonással
        fg_mask = bg_subtractor.apply(gray_frame)
        fg_mask = cv.bitwise_not(fg_mask)  # A maszk invertálása

        if video_mask is None:
            video_mask = np.full_like(fg_mask, 255, dtype=np.uint8)  # Fehér háttérrel kezdődik
        else:
            if fg_mask.sum() / 255 > 1_000_000:
                video_mask = cv.bitwise_and(video_mask, fg_mask)  # Közös mozgásokat tartja meg

    capture.release()  # Videó lezárása

    # Morfológiai műveletek a videó maszkon (simaítás)
    if video_mask is not None:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        video_mask = cv.morphologyEx(video_mask, cv.MORPH_CLOSE, kernel)
        video_mask = cv.morphologyEx(video_mask, cv.MORPH_OPEN, kernel)

    cv.imwrite("rat_movement_mask.png", video_mask)
    cv.destroyAllWindows()
    return video_mask
