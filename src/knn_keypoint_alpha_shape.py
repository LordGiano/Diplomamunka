import cv2
import numpy as np
import os
import time
import remove_grids_from_mask
import create_cleaned_mask
from alpha_shape import draw_alpha_shape, apply_existing_alpha_shape
from knn_mask import process_video
from tqdm import tqdm
import gc


def detect_keypoints_near_mask(frame, mask, vicinity_size=100, detector_type='AKAZE', keypoint_size=None):
    # Konvertálás szürkeárnyalatosra
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Maszk kiterjesztése a környezet méretével
    vicinity_kernel = np.ones((vicinity_size, vicinity_size), np.uint8)
    extended_mask = cv2.dilate(mask, vicinity_kernel, iterations=1)

    # Bináris maszk biztosítása (0 és 255 értékek)
    _, extended_mask = cv2.threshold(extended_mask, 127, 255, cv2.THRESH_BINARY)

    # Detektor kiválasztása
    if detector_type == 'AKAZE':
        detector = cv2.AKAZE_create()
    elif detector_type == 'ORB':
        detector = cv2.ORB_create(nfeatures=1500)
    elif detector_type == 'SIFT':
        detector = cv2.SIFT_create()
    elif detector_type == 'SURF':
        try:
            detector = cv2.xfeatures2d.SURF_create()
        except AttributeError:
            print("SURF nem érhető el ebben az OpenCV verzióban. Váltás ORB-re.")
            detector = cv2.ORB_create(nfeatures=1500)
    else:
        print(f"Ismeretlen detektor típus: {detector_type}. Váltás AKAZE-re.")
        detector = cv2.AKAZE_create()

    # Kulcspontok detektálása a maszk környezetében
    kp, des = detector.detectAndCompute(gray, extended_mask)

    # Kulcspontok méretének felülírása, ha meg van adva
    if keypoint_size is not None:
        for keypoint in kp:
            keypoint.size = keypoint_size

    return kp, des


def filter_keypoints_by_grid_mask(keypoints, grid_mask, grid_vicinity=0):
    # Ha van megadva környezeti méret, kiterjesszük a rácsok maszkját
    if grid_vicinity > 0:
        # Kernel létrehozása a dilatációhoz
        vicinity_kernel = np.ones((grid_vicinity, grid_vicinity), np.uint8)
        # Rács maszk kiterjesztése
        extended_grid_mask = cv2.dilate(grid_mask, vicinity_kernel, iterations=1)
    else:
        extended_grid_mask = grid_mask.copy()

    filtered_kp = []

    for kp in keypoints:
        # Kulcspont koordinátái
        x, y = int(kp.pt[0]), int(kp.pt[1])

        # Ellenőrizzük, hogy a koordináták a kép határain belül vannak-e
        if 0 <= x < extended_grid_mask.shape[1] and 0 <= y < extended_grid_mask.shape[0]:
            # Ellenőrizzük, hogy a pont a kiterjesztett rács maszkjára esik-e
            if extended_grid_mask[y, x] == 0:  # 0 = nem rács
                filtered_kp.append(kp)

    return filtered_kp


def add_keypoints_to_mask(mask, keypoints, radius=5, color=255):
    enhanced_mask = mask.copy()

    # Kulcspontok hozzáadása a maszkhoz
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])

        # Kör rajzolása a kulcspont helyére
        cv2.circle(enhanced_mask, (x, y), radius, color, -1)  # -1: kitöltött kör

    return enhanced_mask


def process_video_with_knn_keypoint_alpha_shape(input_video, output_mask_video, history=300,
                            dist2Threshold=300.0, vicinity_size=100, detector_type='AKAZE',
                            keypoint_size=None, keypoint_radius=5, filter_grid_keypoints=True,
                            grid_vicinity=0):
    """
    Videó feldolgozása csak a maszk videó létrehozásához:
    Eredeti képkocka + továbbfejlesztett maszk (mozgási maszk + kulcspontok)

    Args:
        input_video: Bemeneti videó elérési útja
        output_mask_video: Kimeneti maszk videó elérési útja
        history: KNN háttérkivonó history paramétere
        dist2Threshold: KNN háttérkivonó távolság küszöbértéke
        vicinity_size: A maszk környezetének mérete pixelben
        detector_type: A használni kívánt detektor típusa ('AKAZE', 'ORB', 'SIFT', 'SURF')
        keypoint_size: Kulcspontok méretének felülírása (None esetén az eredeti méret)
        keypoint_radius: A maszkhoz hozzáadott kulcspontok sugara
        filter_grid_keypoints: Rácsra eső kulcspontok kiszűrése (True/False)
        grid_vicinity: A rács környezetének mérete pixelben (csak ha filter_grid_keypoints=True)
    """
    start_time = time.time()

    print("Rács maszkjának létrehozása...")
    base_mask = process_video(input_video)
    cv2.imwrite("base_mask_knn.png", base_mask)

    if base_mask is None:
        print("Hiba: Nem sikerült a mozgásmaszkot létrehozni.")
        return

    cleaned_mask = remove_grids_from_mask.remove_grids_from_mask(base_mask)
    cv2.imwrite("cleaned_mask_knn.png", cleaned_mask)

    grid_mask = cv2.absdiff(base_mask, cleaned_mask)
    _, grid_mask = cv2.threshold(grid_mask, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("grid_mask.png", grid_mask)

    print("Végső maszk generálása...")
    final_mask = create_cleaned_mask.create_cleaned_mask(base_mask, cleaned_mask)
    cv2.imwrite("final_mask_knn.png", final_mask)

    print("Videó feldolgozása és mentése...")

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print(f"Hiba: Nem sikerült a videó megnyitása: {input_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Videó információk: {width}x{height}, {fps} FPS, {frame_count} képkocka")

    output_dir = os.path.dirname(output_mask_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mask_video, fourcc, fps, (width * 2, height))

    bg_subtractor = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=dist2Threshold,
                                                      detectShadows=True)

    # Progress bar inicializálása
    pbar = tqdm(total=frame_count, desc="Képkockák feldolgozása", unit="frame")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mozgási maszk megállapítása
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(frame_gray)

        # Morfológiai műveletek a zaj csökkentéséhez
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        # Bináris maszk biztosítása
        _, motion_mask_binary = cv2.threshold(motion_mask, 127, 255, cv2.THRESH_BINARY)

        # Kulcspont detekció a maszk környékén
        keypoints, descriptors = detect_keypoints_near_mask(
            frame,
            motion_mask_binary,
            vicinity_size=vicinity_size,
            detector_type=detector_type,
            keypoint_size=keypoint_size
        )

        # Rácsra eső kulcspontok eltávolítása (opcionális)
        if filter_grid_keypoints:
            filtered_keypoints = filter_keypoints_by_grid_mask(keypoints, grid_mask, grid_vicinity=grid_vicinity)
        else:
            filtered_keypoints = keypoints

        # Kulcspontok hozzáadása a mozgási maszkhoz
        enhanced_mask = add_keypoints_to_mask(motion_mask_binary, filtered_keypoints, radius=keypoint_radius)

        # 3 csatornás maszk készítése a megjelenítéshez
        enhanced_mask_display = np.stack([enhanced_mask] * 3, axis=2)

        # Fehér pixelek számának ellenőrzése az 50%-os küszöbhöz
        total_pixels = enhanced_mask.shape[0] * enhanced_mask.shape[1]
        white_pixels_count = np.count_nonzero(enhanced_mask > 128)
        white_pixel_ratio = white_pixels_count / total_pixels

        # Alpha shape alkalmazása csak akkor, ha a fehér pixelek aránya <= 50%
        if white_pixel_ratio <= 0.5:
            try:
                _, points, shape = draw_alpha_shape(enhanced_mask, 15, True)
                alpha_shape_frame = apply_existing_alpha_shape(frame, shape)
                alpha_shape_mask = apply_existing_alpha_shape(enhanced_mask_display, shape)
            except Exception as e:
                print(f"Alpha shape hiba frame {frame_idx}: {e}")
                alpha_shape_frame = frame
                alpha_shape_mask = enhanced_mask_display
        else:
            # Ha túl sok fehér pixel van (>50%), kihagyjuk az alpha shape-et
            print(f"Frame {frame_idx}: Túl sok fehér pixel ({white_pixel_ratio:.2%}), alpha shape kihagyva")
            alpha_shape_frame = frame
            alpha_shape_mask = enhanced_mask_display

        # Eredeti képkocka és alpha shape-pel rendelkező maszk egymás mellé helyezése
        #mask_side_by_side = np.hstack((frame, enhanced_mask_display))
        mask_side_by_side = np.hstack((alpha_shape_frame, alpha_shape_mask))

        # Kimeneti videó mentése
        out.write(mask_side_by_side)

        # Folyamatjelző frissítése
        frame_idx += 1
        pbar.update(1)

        # Memória felszabadítása rendszeresen
        if frame_idx % 100 == 0:
            gc.collect()

    # Progress bar bezárása
    pbar.close()

    # Erőforrások felszabadítása
    cap.release()
    out.release()

    # Teljes futási idő kiszámítása és kijelzése
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Videó feldolgozása befejezve!")
    print(f"Kimeneti fájl: {output_mask_video}")
    print(f"Teljes futási idő: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")


def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Teljes szkript futási ideje: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
        return result

    return wrapper


if __name__ == "__main__":
    @time_decorator
    def main():
        input_video_path = r"C:/Programozás/Diplomamunka/videos/Rat5/04.12.01-04.13.41[M][0@0][0].dav"
        output_mask_video_path = r"C:/Programozás/Diplomamunka/feldolgozott_videok/Rat5_KNN_maszk_kulcspontokkal_AKAZE_rács=5.mp4"

        # KNN paraméterek
        history = 200
        dist2Threshold = 100

        # Kulcspont paraméterek
        vicinity_size = 50
        detector_type = 'AKAZE'
        keypoint_size = 8
        keypoint_radius = 5

        # Rács szűrési paraméterek
        filter_grid_keypoints = True
        grid_vicinity = 5

        # Csak maszk videó feldolgozása
        process_video_with_knn_keypoint_alpha_shape(
            input_video_path,
            output_mask_video_path,
            history=history,
            dist2Threshold=dist2Threshold,
            vicinity_size=vicinity_size,
            detector_type=detector_type,
            keypoint_size=keypoint_size,
            keypoint_radius=keypoint_radius,
            filter_grid_keypoints=filter_grid_keypoints,
            grid_vicinity=grid_vicinity
        )


    main()