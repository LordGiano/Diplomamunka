import cv2 as cv
import numpy as np
import os
import random


def inpaint_random_frames(video_path, mask, output_folder="../inpainted_frames", frame_count=5):
    """
    Véletlenszerűen kiválasztott 5 képkockára alkalmazza az inpaintinget, majd elmenti azokat képként.

    :param video_path: A bemeneti videó elérési útja
    :param mask: A rácsokat jelölő maszk (numpy.ndarray)
    :param output_folder: A kimeneti képek fő mappája
    :param frame_count: Az inpaintinget végrehajtó képkockák száma
    """
    # Videó nevének és mappa struktúrájának kinyerése
    video_dir, video_filename = os.path.split(video_path)
    rat_folder = os.path.basename(video_dir)
    output_path = os.path.join(output_folder, rat_folder)
    os.makedirs(output_path, exist_ok=True)

    # Videó beolvasása
    capture = cv.VideoCapture(video_path)
    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    if total_frames < frame_count:
        print(f"Figyelmeztetés: A videó csak {total_frames} képkockát tartalmaz, kevesebb mint {frame_count}.")
        frame_indices = list(range(total_frames))
    else:
        frame_indices = random.sample(range(total_frames), frame_count)  # Véletlenszerű 5 frame kiválasztása

    # Képkockák feldolgozása
    for i, frame_index in enumerate(frame_indices):
        capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = capture.read()
        if not ret:
            print(f"Hiba a {frame_index}. képkocka beolvasásánál.")
            continue

        # Maszk átméretezése, ha szükséges
        if mask.shape != frame.shape[:2]:
            mask_resized = cv.resize(mask, (frame.shape[1], frame.shape[0]))
        else:
            mask_resized = mask

        # --- Maszk finomhangolása ---
        kernel = np.ones((5, 5), np.uint8)  # Kisebb kernel a finomabb eltávolítás érdekében
        cleaned_mask = cv.morphologyEx(mask_resized, cv.MORPH_OPEN, kernel)
        cleaned_mask = cv.morphologyEx(cleaned_mask, cv.MORPH_CLOSE, kernel)
        dilated_mask = cv.dilate(cleaned_mask, kernel, iterations=2) #minél kisebb a kernel, annál inkább látható a rácsok vonalai

        # Inpainting alkalmazása
        inpainted_frame = cv.inpaint(frame, dilated_mask, inpaintRadius=10, flags=cv.INPAINT_TELEA)

        # Képkocka mentése
        output_filename_base = os.path.join(output_path, f"{video_filename}_frame_{frame_index}_base.png")
        output_filename_inpaint = os.path.join(output_path, f"{video_filename}_frame_{frame_index}_inpainted.png")
        cv.imwrite(output_filename_base, frame)
        cv.imwrite(output_filename_inpaint, inpainted_frame)
        print(f"Mentve: {output_filename_inpaint}")

    # Erőforrások felszabadítása
    capture.release()
    cv.destroyAllWindows()

# ideális paraméterek:
# iteratiomn = 1