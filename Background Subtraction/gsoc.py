from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import os
import datetime

# Argumentum parser az input videó útvonalához
parser = argparse.ArgumentParser(description='GSOC háttérkivonási módszer alkalmazása.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.',
                    default='../videos/Rat1/08.07.02-08.09.03[M][0@0][0].dav')
parser.add_argument('--mc', type=bool, help='Use motion compensation in GSOC.', default=True)
parser.add_argument('--nSamples', type=int, help='Number of samples in GSOC.', default=20)
parser.add_argument('--replaceRate', type=float, help='Replace rate for GSOC.', default=0.003)
parser.add_argument('--propagationRate', type=float, help='Propagation rate for GSOC.', default=0.01)
parser.add_argument('--hitsThreshold', type=int, help='Hits threshold for GSOC.', default=32)
parser.add_argument('--alpha', type=float, help='Alpha parameter for GSOC.', default=0.01)
parser.add_argument('--beta', type=float, help='Beta parameter for GSOC.', default=0.0022)
parser.add_argument('--blinkingSupressionDecay', type=float, help='Blinking suppression decay in GSOC.', default=0.1)
parser.add_argument('--blinkingSupressionMultiplier', type=float, help='Blinking suppression multiplier in GSOC.', default=0.1)
args = parser.parse_args()

# GSOC háttérkivonási módszer létrehozása a paraméterekkel
backSub = cv.bgsegm.createBackgroundSubtractorGSOC(mc=int(args.mc), nSamples=args.nSamples, replaceRate=args.replaceRate,
                                                   propagationRate=args.propagationRate, hitsThreshold=args.hitsThreshold,
                                                   alpha=args.alpha, beta=args.beta, blinkingSupressionDecay=args.blinkingSupressionDecay,
                                                   blinkingSupressionMultiplier=args.blinkingSupressionMultiplier)

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
            save_screenshot("GSOC", last_frame, last_filtered_fgMask, frame_index)
    elif keyboard == 27 or keyboard == ord('q') or keyboard == ord('Q'):  # Kilépés
        break


# Felszabadítja az erőforrásokat és bezárja az ablakokat
capture.release()
cv.destroyAllWindows()
