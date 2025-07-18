import numpy as np
import cv2

def create_cleaned_mask(mask: np.ndarray, cleaned_mask: np.ndarray):
    result_mask = mask.copy()
    result_mask[cleaned_mask == 255] = 0

    cv2.imwrite("cleaned_mask.png", result_mask)
    return result_mask
