import cv2 as cv
import numpy as np
import os
import dilated_mask
import time
from tqdm import tqdm
import create_mask
import remove_grids_from_mask
import create_cleaned_mask


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


def inpaint_frame(frame, mask, kernel_size=(3, 3), dilation_iterations=2, inpaint_radius=7, method=cv.INPAINT_TELEA):
    # Ellenőrizzük, hogy a maszk mérete megegyezik-e a képkocka méretével
    if mask.shape != frame.shape[:2]:
        # Ha a maszk mérete eltér, átméretezzük
        resized_mask = cv.resize(mask, (frame.shape[1], frame.shape[0]))
        print("Figyelmeztetés: A maszk átméretezve a képkocka méretére!")
    else:
        resized_mask = mask.copy()

    # Maszk finomhangolása a dilated_mask modullal
    dilated = dilated_mask.create_dilated_mask(resized_mask, kernel_size=kernel_size, iterations=dilation_iterations)

    inpainted_frame = cv.inpaint(frame, dilated, inpaintRadius=inpaint_radius, flags=method)

    return inpainted_frame


def create_knn_with_grid_inpainting_video(video_path, output_path=None,
                                          history=400, dist2Threshold=300.0, detectShadows=False,
                                          kernel_size=(3, 3), dilation_iterations=2, inpaint_radius=7,
                                          method=cv.INPAINT_TELEA):

    if not os.path.exists(video_path):
        raise ValueError(f"A fájl nem található: {video_path}")

    print(f"Feldolgozás kezdete: {video_path}")

    # Statikus rácsmaszk létrehozása (ez a teljes videó alapján készül)
    print("1. Mozgásmaszk létrehozása...")
    base_mask = create_mask.process_video(video_path)
    if base_mask is None:
        raise RuntimeError("Hiba történt a mozgásmaszk létrehozásakor!")

    # Rácsok eltávolítása
    print("2. Rácsok eltávolítása...")
    cleaned_mask = remove_grids_from_mask.remove_grids_from_mask(base_mask)

    # Végső rácsmaszk generálása (ez a statikus maszk az inpaintinghez)
    print("3. Végső rácsmaszk generálása...")
    grid_mask = create_cleaned_mask.create_cleaned_mask(base_mask, cleaned_mask)

    if output_path is None:
        video_dir, video_filename = os.path.split(video_path)
        rat_folder = os.path.basename(video_dir)  # Pl: "Rat5"

        method_name = "TELEA" if method == cv.INPAINT_TELEA else "NS"
        params_str = f"(h={history}, d={dist2Threshold}, i={inpaint_radius}, m={method_name})"

        output_folder = os.path.join("../inpainted_videos", rat_folder)
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder,
                                   video_filename.replace(".dav", f" - KNN_Grid_Inpainting {params_str}.mp4"))
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"Hiba: Nem sikerült megnyitni a videót: {video_path}")
        return None

    fps = int(capture.get(cv.CAP_PROP_FPS))
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    print(f"Videó információk: {width}x{height}, {fps} FPS, {frame_count} képkocka")
    print(f"KNN paraméterek: history={history}, dist2Threshold={dist2Threshold}, detectShadows={detectShadows}")
    print(f"Inpainting paraméterek: kernel_size={kernel_size}, dilation_iterations={dilation_iterations}, inpaint_radius={inpaint_radius}")

    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # MP4 formátumhoz megfelelő kodek
    out = cv.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    # KNN háttérkivonási módszer létrehozása
    backSub = cv.createBackgroundSubtractorKNN(history=history,
                                               dist2Threshold=dist2Threshold,
                                               detectShadows=detectShadows)

    print("4. KNN mozgásdetektálás és rácsmaszk inpainting...")

    # --- Videó feldolgozása progress bar-ral ---
    completed_frames = 0

    # Progress bar inicializálása
    pbar = tqdm(total=frame_count, desc="Képkockák feldolgozása", unit="frame")

    while True:
        ret, frame = capture.read()
        if not ret:
            break  # Kilépés, ha nincs több képkocka

        # KNN háttérkivonás alkalmazása a mozgás detektálásához
        fg_mask = backSub.apply(frame)

        # Mozgási maszk szűrése (zajcsökkentés)
        # Só-bors zaj eltávolítása
        filtered_fg_mask = cv.medianBlur(fg_mask, 5)

        # Morfológiai műveletek további tisztításhoz
        kernel = np.ones((3, 3), np.uint8)
        filtered_fg_mask = cv.morphologyEx(filtered_fg_mask, cv.MORPH_OPEN, kernel, iterations=1)
        filtered_fg_mask = cv.morphologyEx(filtered_fg_mask, cv.MORPH_CLOSE, kernel, iterations=1)

        inpainted_frame = inpaint_frame(filtered_fg_mask, grid_mask,
                                        kernel_size=kernel_size,
                                        dilation_iterations=dilation_iterations,
                                        inpaint_radius=inpaint_radius,
                                        method=method)

        # Összehasonlító kép létrehozása (KNN maszk bal oldalon, inpaintingelt jobb oldalon)
        inpainted_frame_color = cv.cvtColor(inpainted_frame, cv.COLOR_GRAY2BGR)
        comparison_frame = np.hstack((frame, inpainted_frame_color))

        out.write(comparison_frame)
        completed_frames += 1

        pbar.update(1)

    pbar.close()

    capture.release()
    out.release()

    print(f"Az KNN + rácsmaszk inpainting videó mentve: {output_path}")
    print(f"Feldolgozott képkockák: {completed_frames}")

    return output_path


def process_video_with_knn_inpainting(video_path: str, output_path: str = None, **kwargs):
    return create_knn_with_grid_inpainting_video(video_path, output_path, **kwargs)


if __name__ == "__main__":
    @time_decorator
    def main():
        # Videó feldolgozása KNN mozgásdetektálás + statikus rácsmaszk inpainting kombinációval
        input_video_path = r"videos\Rat4\04.12.02-04.14.21[M][0@0][0].dav"
        output_video_path = r"processd_video.mp4"

        # KNN paraméterek
        history = 400
        dist2Threshold = 300.0
        detectShadows = False

        # Inpainting paraméterek
        kernel_size = (3, 3)
        dilation_iterations = 2
        inpaint_radius = 7
        method = cv.INPAINT_TELEA

        process_video_with_knn_inpainting(
            input_video_path,
            output_video_path,
            history=history,
            dist2Threshold=dist2Threshold,
            detectShadows=detectShadows,
            kernel_size=kernel_size,
            dilation_iterations=dilation_iterations,
            inpaint_radius=inpaint_radius,
            method=method
        )

    main()