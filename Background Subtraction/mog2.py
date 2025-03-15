from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import os
import datetime

# Argumentum parser az input videó útvonalához
parser = argparse.ArgumentParser(description='MOG2 háttérkivonási módszer alkalmazása.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.',
                    default='../videos/Rat1/08.07.02-08.09.03[M][0@0][0].dav')
parser.add_argument('--history', type=int, help='History parameter for MOG2.', default=1500) #500
parser.add_argument('--varThreshold', type=float, help='Variance threshold for MOG2.', default=25) #16
parser.add_argument('--detectShadows', type=bool, help='Detect shadows in MOG2.', default=False) #True
args = parser.parse_args()

# MOG2 háttérkivonási módszer létrehozása a paraméterekkel
backSub = cv.createBackgroundSubtractorMOG2(history=args.history, varThreshold=args.varThreshold,
                                            detectShadows=args.detectShadows)

# Videó beolvasása
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

# Képernyőkép készítési funkció
def save_screenshot(window_name, frame, filtered_frame, frame_index):
    video_path = args.input  # Videó teljes elérési útja
    video_filename = os.path.basename(video_path)  # Videófájl neve kiterjesztéssel
    video_name = os.path.splitext(video_filename)[0]  # Kiterjesztés eltávolítása

    # Kivonjuk az időtartamot a fájlnévből (első '[' előtti rész)
    parts = video_name.split('[')[0]  # Példa: "08.07.02-08.09.03"

    # RatX az elérési útból (utolsó előtti mappa neve)
    rat_folder = os.path.basename(os.path.dirname(video_path))  # Példa: "Rat1"

    # Screenshot mappa struktúra generálása
    directory = os.path.join(window_name, rat_folder)  # Pl: "CNT/Rat1"

    # A fájlnév generálása
    filename = os.path.join(directory, f"{parts}_FRAME_{frame_index}_combined.png")  # Kép mentése ezen belül

    # Ellenőrizzük a színcsatornákat, és konvertáljuk, ha szükséges
    if len(frame.shape) == 3 and len(filtered_frame.shape) == 2:  # Ha az egyik szürkeárnyalatos
        filtered_frame = cv.cvtColor(filtered_frame, cv.COLOR_GRAY2BGR)  # Konvertáljuk RGB-re

    # Képek összeillesztése
    combined_frame = cv.hconcat([frame, filtered_frame])  # Az eredeti és a szűrt kép egymás mellé helyezése

    # Mappák létrehozása
    os.makedirs(directory, exist_ok=True)

    # Képernyőkép mentése
    success = cv.imwrite(filename, combined_frame)
    if success:
        print(f"Képernyőkép elmentve: {filename}")
    else:
        print(f"Hiba: Nem sikerült a képernyőkép mentése: {filename}")


# Állapotváltozó a lejátszás/szünet kezeléséhez
paused = False

# Tárolók az utolsó frame-ekhez
last_frame = None
last_fgMask = None
last_filtered_fgMask = None

while True:
    # Ha nincs szünet, akkor olvassunk be egy új képkockát
    if not paused:
        ret, frame = capture.read()
        if frame is None:
            break

        # Aktuális frame index lekérése
        frame_index = int(capture.get(cv.CAP_PROP_POS_FRAMES))  # Frame index

        # Háttérkivonás alkalmazása
        fgMask = backSub.apply(frame)

        # Só-bors zajszűrés
        filtered_fgMask = cv.medianBlur(fgMask, 5)

        # Morfológiai szűrés (kis zajok és rácsok eltávolítása)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # Kicsi, 3x3-as ellipszis kernel
        filtered_fgMask = cv.morphologyEx(filtered_fgMask, cv.MORPH_OPEN,
                                          kernel)  # Nyitás a vékony zajok eltávolítására
        filtered_fgMask = cv.morphologyEx(filtered_fgMask, cv.MORPH_CLOSE, kernel)  # Zárás a kisebb lyukak kitöltésére

        # Az aktuális frame-ek tárolása a szünethez
        last_frame = frame.copy()
        last_fgMask = fgMask.copy()
        last_filtered_fgMask = filtered_fgMask.copy()

    # Ha szünetel a program, a tárolt frame-eket használjuk
    if last_frame is not None:
        cv.imshow('Original Video', last_frame)
    if last_fgMask is not None:
        cv.imshow('Foreground Mask', last_fgMask)
    if last_filtered_fgMask is not None:
        cv.imshow('Processed Video', last_filtered_fgMask)

    # Billentyűfigyelés
    keyboard = cv.waitKey(30) & 0xFF
    if keyboard == ord(' '):  # Space megnyomása
        paused = not paused  # Lejátszás/szünet váltás
    elif keyboard == ord('x') or keyboard == ord('X'):  # Screenshot mentése
        if last_filtered_fgMask is not None:
            save_screenshot("MOG2", last_frame, last_filtered_fgMask, frame_index)
    elif keyboard == 27 or keyboard == ord('q') or keyboard == ord('Q'):  # Kilépés
        break


# Felszabadítja az erőforrásokat és bezárja az ablakokat
capture.release()
cv.destroyAllWindows()
