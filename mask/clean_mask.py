import cv2 as cv
import numpy as np

# --- Maszk beolvasása ---
mask_path = "grid_mask.png"
mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

if mask is None:
    print("Hiba: Nem sikerült beolvasni a maszkot!")
    exit(1)

# --- Maszk mérete ---
h, w = mask.shape

# --- LÉPÉS 1: Zajcsökkentés és élek finomítása ---
# Kisebb fehér zajok eltüntetése, hogy ne kapcsolódjanak össze a széleken lévő foltokkal
blurred = cv.GaussianBlur(mask, (5, 5), 0)

# --- LÉPÉS 2: A maszk széleinek levágása, hogy csak a középső rész maradjon ---
crop_margin = 50  # Beállítható érték: a széleket ennyivel vágjuk le
cropped_mask = np.zeros_like(mask)
cropped_mask[crop_margin:h - crop_margin, crop_margin:w - crop_margin] = \
    mask[crop_margin:h - crop_margin, crop_margin:w - crop_margin]

# --- LÉPÉS 3: Morfológiai műveletek a rácsok megőrzésére ---
kernel = np.ones((5, 5), np.uint8)
opened_mask = cv.morphologyEx(cropped_mask, cv.MORPH_OPEN, kernel)  # Kis zajok eltávolítása
closed_mask = cv.morphologyEx(opened_mask, cv.MORPH_CLOSE, kernel)  # Lyukak betömése

# --- LÉPÉS 4: Kontúrok keresése és szétválasztása ---
contours, _ = cv.findContours(closed_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

racsok = np.zeros_like(mask)  # A rácsokat tároljuk itt

for cnt in contours:
    area = cv.contourArea(cnt)

    # Megtartjuk a közepes méretű régiókat (a rácsokat)
    if 1000 < area < 50000:
        cv.drawContours(racsok, [cnt], -1, (255), thickness=cv.FILLED)

# --- LÉPÉS 5: Végleges maszk mentése ---
output_path = "/mnt/data/cleaned_mask.png"
cv.imwrite(output_path, racsok)
print(f"Tisztított maszk elmentve: {output_path}")

# --- Megjelenítés ---
cv.imshow("Eredeti Maszk", mask)
cv.imshow("Levágott Maszk", cropped_mask)
cv.imshow("Rácsok (Megmaradnak)", racsok)
cv.imshow("Végső Tisztított Maszk", racsok)
cv.waitKey(0)
cv.destroyAllWindows()
