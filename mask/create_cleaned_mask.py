import cv2
import numpy as np
import matplotlib.pyplot as plt

# Képek betöltése
grid_mask_path = "grid_mask.png"  # Módosítsd az útvonalat, ha szükséges
cleaned_mask_path = "cleaned_mask.png"

# Eredeti és tisztított maszk betöltése
grid_mask = cv2.imread(grid_mask_path, cv2.IMREAD_GRAYSCALE)
cleaned_mask = cv2.imread(cleaned_mask_path, cv2.IMREAD_GRAYSCALE)

# Ellenőrizzük, hogy mindkét kép sikeresen betöltődött-e
if grid_mask is None or cleaned_mask is None:
    raise ValueError("Nem sikerült betölteni az egyik képet.")

# Azokon a helyeken, ahol a cleaned_mask fehér (255), a grid_mask feketévé válik (0)
result_mask = grid_mask.copy()
result_mask[cleaned_mask == 255] = 0

# Eredmény mentése
cv2.imwrite("filtered_grid_mask.png", result_mask)
