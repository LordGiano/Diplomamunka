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


def process_video_with_base_inpainting(video_path: str, output_path: str = None):
    if not os.path.exists(video_path):
        raise ValueError(f"A fájl nem található: {video_path}")

    print(f"Feldolgozás kezdete: {video_path}")

    # Mozgásmaszk létrehozása
    print("Mozgásmaszk létrehozása...")
    base_mask = create_mask.process_video(video_path)
    if base_mask is None:
        raise RuntimeError("Hiba történt a mozgásmaszk létrehozásakor!")

    # Rácsok eltávolítása
    print("Rácsok eltávolítása...")
    cleaned_mask = remove_grids_from_mask.remove_grids_from_mask(base_mask)

    # Végső maszk generálása
    print("Végső maszk generálása...")
    final_mask = create_cleaned_mask.create_cleaned_mask(base_mask, cleaned_mask)

    print("Összehasonlító videó létrehozása...")
    output_video_path = create_video(video_path, final_mask, output_path=output_path)

    print(f"Feldolgozás kész. Kimeneti videó: {output_video_path}")
    return output_video_path


def create_video(video_path, mask, output_path=None, kernel_size=(3, 3), dilation_iterations=2,
                            inpaint_radius=7, method=cv.INPAINT_TELEA):
    if output_path is None:
        video_dir, video_filename = os.path.split(video_path)
        rat_folder = os.path.basename(video_dir)

        method_name = "TELEA" if method == cv.INPAINT_TELEA else "NS"
        params_str = f"(k={kernel_size[0]}, d={dilation_iterations}, i={inpaint_radius}, m={method_name})"

        output_folder = os.path.join("../inpainted_videos", rat_folder)
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder,
                                   video_filename.replace(".dav", f" - Comparison {method_name} {params_str}.mp4"))
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

    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # MP4 formátumhoz megfelelő kodek
    out = cv.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    completed_frames = 0

    pbar = tqdm(total=frame_count, desc="Képkockák feldolgozása", unit="frame")

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        inpainted_frame = inpaint_frame(frame, mask,
                                        kernel_size=kernel_size,
                                        dilation_iterations=dilation_iterations,
                                        inpaint_radius=inpaint_radius,
                                        method=method)

        # Összehasonlító kép létrehozása (eredeti bal oldalon, inpaintingelt jobb oldalon)
        comparison_frame = np.hstack((frame, inpainted_frame))

        out.write(comparison_frame)
        completed_frames += 1

        pbar.update(1)

    pbar.close()

    capture.release()
    out.release()

    print(f"Az összehasonlító videó mentve: {output_path}")
    print(f"Feldolgozott képkockák: {completed_frames}")

    return output_path


if __name__ == "__main__":
    @time_decorator
    def main():
        input_video_path = r"C:videos\Rat4\04.12.02-04.14.21[M][0@0][0].dav"
        output_video_path = r"inpainting.mp4"

        process_video_with_base_inpainting(input_video_path, output_video_path)


    main()