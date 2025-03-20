import cv2 as cv
import numpy as np
import os

def inpaint_video(video_path, mask, output_root="../inpainted_videos"):
    """
    Inpainting elvégzése egy videóra, majd a feldolgozott videó mentése az adott patkány mappába.

    :param video_path: A bemeneti videó elérési útja
    :param mask: A rácsokat jelölő maszk (numpy.ndarray)
    :param output_root: A kimeneti videók fő mappája
    """
    # Videó nevének és mappa struktúrájának kinyerése
    video_dir, video_filename = os.path.split(video_path)
    rat_folder = os.path.basename(video_dir)  # Pl: "Rat5"

    # Kimeneti mappa létrehozása
    output_folder = os.path.join(output_root, rat_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Kimeneti fájl neve és elérési útja
    output_video_path = os.path.join(output_folder, video_filename.replace(".dav", " - NS (inpaintRadius-15).mp4"))

    # --- Videó beolvasása ---
    capture = cv.VideoCapture(video_path)

    # --- Ellenőrizzük a maszk méretét ---
    ret, frame = capture.read()
    if frame is None:
        print("Hiba: Nem sikerült beolvasni a videót!")
        return

    # Ha a maszk mérete eltér, átméretezzük
    if mask.shape != frame.shape[:2]:
        print("Maszk átméretezve!")
        mask = cv.resize(mask, (frame.shape[1], frame.shape[0]))

    # --- Maszk finomhangolása ---
    #kernel = np.ones((7, 7), np.uint8)  # Kisebb kernel a finomabb eltávolítás érdekében
    #cleaned_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    #cleaned_mask = cv.morphologyEx(cleaned_mask, cv.MORPH_CLOSE, kernel)
    #dilated_mask = cv.dilate(cleaned_mask, np.ones((5, 5), np.uint8), iterations=1)

    # --- Videó író beállítása ---
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # MP4 formátumhoz megfelelő kodek
    fps = int(capture.get(cv.CAP_PROP_FPS))
    frame_size = (frame.shape[1], frame.shape[0])
    out = cv.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Összes képkocka számolása
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    completed_frames = 0

    # --- Videó feldolgozása ---
    while True:
        ret, frame = capture.read()
        if not ret:
            break  # Kilépés, ha nincs több képkocka

        # --- Inpainting csak a tisztított maszkon ---
        #inpainted_frame = cv.inpaint(frame, dilated_mask, inpaintRadius=10, flags=cv.INPAINT_NS)
        inpainted_frame = cv.inpaint(frame, mask, inpaintRadius=15, flags=cv.INPAINT_NS)

        # --- Képkocka mentése az új videóba ---
        out.write(inpainted_frame)
        completed_frames += 1

        # Képkocka számláló kiírása
        print(f"{completed_frames}/{frame_count}")

    # --- Erőforrások felszabadítása ---
    capture.release()
    out.release()
    cv.destroyAllWindows()

    print(f"Az inpaintingelt videó mentve: {output_video_path}")
