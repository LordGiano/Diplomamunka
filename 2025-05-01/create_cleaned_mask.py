import numpy as np
import cv2

def create_cleaned_mask(mask: np.ndarray, cleaned_mask: np.ndarray):
    """
    A tisztított maszk alapján eltávolítja a rácsokat az eredeti maszkról.

    :param mask: Az eredeti mozgásmaszk numpy tömbként.
    :param cleaned_mask: A tisztított maszk numpy tömbként.
    :return: A végső maszk numpy tömbként.
    """
    result_mask = mask.copy()
    result_mask[cleaned_mask == 255] = 0

    cv2.imwrite("cleaned_mask.png", result_mask)  # ⬅️ Kimenthető, ha szükséges
    return result_mask
