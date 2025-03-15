import cv2
import numpy as np
import matplotlib.pyplot as plt

# Kép betöltése
image_path = "grid_mask.png"  # Módosítsd az útvonalat, ha szükséges
mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ellenőrizzük, hogy a kép sikeresen betöltődött-e
if mask is None:
    raise ValueError("A kép nem található vagy nem sikerült betölteni.")

def apply_morphological_operations(mask):
    """
    Alkalmaz morfológiai műveleteket (nyitás és zárás) a maszk tisztítására.
    """
    kernel = np.ones((9, 9), np.uint8)  # Kernel méret a morfológiai műveletekhez

    # Nyitás: Eltávolítja a kisebb fehér zajokat
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Zárás: Kitölti a kisebb fekete lyukakat
    cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return cleaned_mask

# 1. Morfológiai műveletek alkalmazása
cleaned_mask = apply_morphological_operations(mask)

# Megjelenítés
plt.figure(figsize=(8, 6))
plt.imshow(cleaned_mask, cmap='gray')
plt.title("Morfológiai műveletekkel tisztított maszk")
plt.axis("off")
plt.show()

# Ha el szeretnéd menteni a tisztított maszkot:
cv2.imwrite("cleaned_mask.png", cleaned_mask)
