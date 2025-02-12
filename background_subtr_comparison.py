from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import datetime

# Argumentum parser az input videó útvonalához
parser = argparse.ArgumentParser(description='Háttér kivonás összehasonlítása.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.',
                    default='videos\\Rat1\\08.07.02-08.09.03[M][0@0][0].dav')
args = parser.parse_args()

# Háttérkivonási módszerek létrehozása
methods = {
    'MOG': cv.bgsegm.createBackgroundSubtractorMOG(),
    'MOG2': cv.createBackgroundSubtractorMOG2(),
    'KNN': cv.createBackgroundSubtractorKNN(),
    'CNT': cv.bgsegm.createBackgroundSubtractorCNT(),
    'GMG': cv.bgsegm.createBackgroundSubtractorGMG(),
    'LSBP': cv.bgsegm.createBackgroundSubtractorLSBP(),
    'GSOC': cv.bgsegm.createBackgroundSubtractorGSOC()
}

# Videó beolvasása
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)


# Képernyőkép készítési funkció
def save_screenshot(window_name, frame):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{window_name}_screenshot_{timestamp}.png"
    cv.imwrite(filename, frame)
    print(f"Képernyőkép elmentve: {filename}")


# Állapotváltozó a lejátszás/szünet kezeléséhez
paused = False

# Fő ciklus a frame-ek feldolgozásához
while True:
    # Ha nincs szünet, akkor olvassunk be egy új képkockát
    if not paused:
        ret, frame = capture.read()
        if frame is None:
            break

    # Tároló az egyes eljárások eredményeinek
    results = []
    frame_height, frame_width = frame.shape[:2]
    display_size = (frame_width // 3, frame_height // 3)  # Minden részablak mérete egységes lesz

    # Minden háttérkivonási eljárásra elvégezzük a kivonást
    for name, backSub in methods.items():
        fgMask = backSub.apply(frame)

        # Az eljárás nevét ráírjuk a maszkra
        cv.putText(fgMask, name, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255), 2)

        # Eredmény átméretezése az egységes megjelenítéshez
        resized_fgMask = cv.resize(fgMask, display_size)

        # Konvertálás szürkeárnyalatos 3 csatornás képpé
        resized_fgMask_colored = cv.cvtColor(resized_fgMask, cv.COLOR_GRAY2BGR)
        results.append(resized_fgMask_colored)

    # Az eredeti videó képkockáját is átméretezzük és hozzáadjuk a listához a középső helyen
    resized_original = cv.resize(frame, display_size)
    cv.putText(resized_original, 'Original', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Rácsszerű elrendezés előkészítése 3x3-as mátrixban
    rows = []
    row_size = 3
    index = 0

    for i in range(3):
        row_images = []
        for j in range(3):
            if i == 1 and j == 1:
                # Középső helyre az eredeti képkockaq
                row_images.append(resized_original)
            else:
                # További helyekre a háttérkivonási eljárások eredményei
                if index < len(results):
                    row_images.append(results[index])
                    index += 1
                else:
                    # Ha kevesebb, mint 8 háttérkivonási eredmény van, kipótlás fekete (3 csatornás) képpel
                    empty_image = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
                    row_images.append(empty_image)

        row = np.hstack(row_images)
        rows.append(row)

    # Összes sor egyesítése függőlegesen
    combined_frame = np.vstack(rows)

    # Megjelenítjük az összesített ablakot, ahol az egyes eljárások eredményei láthatók
    cv.imshow('Background Subtraction Comparison', combined_frame)

    # Billentyűfigyelés a lejátszás/szünet, képernyőkép készítés és kilépéshez
    keyboard = cv.waitKey(30) & 0xFF
    if keyboard == ord(' '):  # Szóköz megnyomása
        paused = not paused  # Lejátszás/szünet váltás
    elif keyboard == ord('x') or keyboard == ord('X'):
        save_screenshot("Background_Subtraction_Comparison", combined_frame)
    elif keyboard == 27 or keyboard == ord('q') or keyboard == ord('Q'):  # ESC vagy Q a kilépéshez
        break

# Felszabadítja az erőforrásokat és bezárja az ablakokat
capture.release()
cv.destroyAllWindows()
