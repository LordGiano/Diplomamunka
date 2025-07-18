import cv2
import numpy as np

def remove_grids_from_mask(mask: np.ndarray):
    kernel = np.ones((9, 9), np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imwrite("cleaned_mask.png", cleaned_mask)
    return cleaned_mask
