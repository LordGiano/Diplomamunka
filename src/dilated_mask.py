import cv2 as cv
import numpy as np
import os


def create_dilated_mask(mask, kernel_size=(3, 3), iterations=2):
    kernel = np.ones(kernel_size, np.uint8)

    # Maszk tisztítása morfológiai műveletekkel
    cleaned_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)  # Kisebb zajok eltávolítása
    cleaned_mask = cv.morphologyEx(cleaned_mask, cv.MORPH_CLOSE, kernel)  # Kisebb lyukak betöltése

    dilated_mask = cv.dilate(cleaned_mask, kernel, iterations=iterations)

    return dilated_mask