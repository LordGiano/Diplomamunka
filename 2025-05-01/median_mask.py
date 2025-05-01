import cv2
import numpy as np
import argparse
import os


def create_movement_mask(input_video, background_image, threshold=30):
    # Háttérkép betöltése
    background = cv2.imread(background_image)
    if background is None:
        print(f"Hiba: Nem sikerült betölteni a háttérképet: {background_image}")
        return None

    # Háttérkép szürkeskálás konvertálása
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # Videó betöltése
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Hiba: Nem sikerült a videó megnyitása: {input_video}")
        return None

    # Kezdeti fehér maszk létrehozása
    ret, first_frame = cap.read()
    if not ret:
        print("Hiba: Nem sikerült olvasni a videóból.")
        return None

    height, width = first_frame.shape[:2]
    movement_mask = np.full((height, width), 255, dtype=np.uint8)  # Fehér maszk inicializálása

    print("Mozgási maszk létrehozása...")
    frame_count = 0

    # Videó feldolgozása
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Visszatekerés az elejére
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Kép szürkeskálás konvertálása
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Abszolút különbség a háttérképhez képest
        diff = cv2.absdiff(gray_frame, background_gray)

        # Küszöbölés a zajelnyomáshoz
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Morfológiai műveletek a zaj csökkentésére
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # A mozgási területek befeketítése a maszkban
        # Itt invertáljuk a küszöbölt képet, hogy ahol mozgás van, ott fekete legyen
        movement_area = cv2.bitwise_not(thresh)

        # Frissítjük a maszkot: ahol mozgás van (movement_area fekete), ott a maszk is fekete lesz
        movement_mask = cv2.bitwise_and(movement_mask, movement_area)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  - {frame_count} képkocka feldolgozva")

    # Erőforrások felszabadítása
    cap.release()

    # Végső morfológiai simítások a maszkon
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    movement_mask = cv2.morphologyEx(movement_mask, cv2.MORPH_CLOSE, kernel)
    movement_mask = cv2.morphologyEx(movement_mask, cv2.MORPH_OPEN, kernel)

    return movement_mask


def main():
    # Argumentum parser beállítása
    parser = argparse.ArgumentParser(description='Mozgási maszk készítése videóból és háttérképből')
    parser.add_argument('--input', type=str,
                        help='Bemeneti videó elérési útja',
                        default='../videos/Rat4/04.18.08-04.20.13[M][0@0][0].dav')
    parser.add_argument('--background', type=str,
                        help='Háttérkép elérési útja',
                        default='background_FINAL.png')
    parser.add_argument('--output', type=str,
                        help='Kimeneti maszk elérési útja',
                        default='rat_movement_mask.png')
    parser.add_argument('--threshold', type=int,
                        help='Küszöbérték a mozgásdetektáláshoz',
                        default=40)

    args = parser.parse_args()

    # Mozgási maszk létrehozása
    movement_mask = create_movement_mask(args.input, args.background, args.threshold)

    if movement_mask is not None:
        # Maszk mentése
        cv2.imwrite(args.output, movement_mask)
        print(f"Mozgási maszk elmentve: {args.output}")

        # Opcionális: Maszk megjelenítése
        cv2.imshow('Mozgási maszk', movement_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()