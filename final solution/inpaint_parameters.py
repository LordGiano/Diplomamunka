import random
import cv2 as cv
import numpy as np
import os
import itertools

# Optimalizált paraméterek
opening_closing_kernels = [3, 5, 7, 9]
dilate_kernels = [3, 5, 7]
dilate_iterations = [1, 2]
inpaint_radii = [3, 5, 7, 9, 11]
inpaint_methods = [cv.INPAINT_TELEA, cv.INPAINT_NS]

# Kimeneti mappa
output_folder = "../inpainted_test_results"
os.makedirs(output_folder, exist_ok=True)

# Paraméterkombinációk generálása
param_combinations = list(
    itertools.product(opening_closing_kernels, dilate_kernels, dilate_iterations, inpaint_radii, inpaint_methods))


def process_frame(frame, mask, params):
    opening_kernel_size, dilate_kernel_size, dilation_iter, inpaint_radius, inpaint_method = params

    # Kernel létrehozása
    opening_kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)

    # Morfológiai műveletek
    cleaned_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, opening_kernel)
    cleaned_mask = cv.morphologyEx(cleaned_mask, cv.MORPH_CLOSE, opening_kernel)
    dilated_mask = cv.dilate(cleaned_mask, dilate_kernel, iterations=dilation_iter)

    # Inpainting végrehajtása (morfológiailag módosított maszkkal)
    inpainted_frame = cv.inpaint(frame, dilated_mask, inpaintRadius=inpaint_radius, flags=inpaint_method)
    return inpainted_frame


def inpaint_test(video_path, mask):
    capture = cv.VideoCapture(video_path)
    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    frame_index = random.randint(0, total_frames - 1)
    capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = capture.read()

    if not ret:
        print("Hiba a videó olvasásánál!")
        return

    # Maszk átméretezése, ha szükséges
    if mask.shape != frame.shape[:2]:
        mask = cv.resize(mask, (frame.shape[1], frame.shape[0]))

    # Inpainting az eredeti maszkkal (módosítás nélkül)
    for method in inpaint_methods:
        method_name = "TELEA" if method == cv.INPAINT_TELEA else "NS"
        raw_inpainted_frame = cv.inpaint(frame, mask, inpaintRadius=10, flags=method)
        raw_filename = f"frame_{frame_index}_raw_{method_name}.png"
        raw_output_path = os.path.join(output_folder, raw_filename)
        cv.imwrite(raw_output_path, raw_inpainted_frame)
        print(f"Mentve (eredeti maszk): {raw_output_path}")

    # Inpainting az összes paraméterkombinációval
    for params in param_combinations:
        inpainted_frame = process_frame(frame, mask, params)

        # Paraméterek mentése fájlnévben
        opening, dilate, iter, radius, method = params
        method_name = "TELEA" if method == cv.INPAINT_TELEA else "NS"
        filename = f"frame_{frame_index}_op{opening}_di{dilate}_it{iter}_r{radius}_{method_name}.png"
        output_path = os.path.join(output_folder, filename)
        cv.imwrite(output_path, inpainted_frame)
        print(f"Mentve: {output_path}")

    capture.release()
    cv.destroyAllWindows()
