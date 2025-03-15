import cv2 as cv
import numpy as np

# --- Beállítások ---
video_path = "../videos/Rat2/05.13.23-05.30.35[M][0@0][0].dav"  # A videó elérési útja
mask_path = "filtered_grid_mask.png"  # A rácsokat jelölő maszk

# --- Videó és maszk beolvasása ---
capture = cv.VideoCapture(video_path)
grid_mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

# --- Ellenőrizzük a maszk méretét ---
ret, frame = capture.read()
if frame is None:
    print("Hiba: Nem sikerült beolvasni a videót!")
    exit(1)

# Ha a maszk mérete eltér, átméretezzük
if grid_mask.shape != frame.shape[:2]:
    grid_mask = cv.resize(grid_mask, (frame.shape[1], frame.shape[0]))

# --- Maszk finomhangolása ---
kernel = np.ones((7, 7), np.uint8)  # Kisebb kernel a finomabb eltávolítás érdekében

# Kis fehér zajok eltávolítása
cleaned_mask = cv.morphologyEx(grid_mask, cv.MORPH_OPEN, kernel)

# Kis lyukak betömése a maszkon (opcionális)
cleaned_mask = cv.morphologyEx(cleaned_mask, cv.MORPH_CLOSE, kernel)

# A rácsok körüli terület növelése, hogy az inpainting jobban működjön
dilated_mask = cv.dilate(cleaned_mask, np.ones((5, 5), np.uint8), iterations=1)

# --- Videó feldolgozása ---
while True:
    ret, frame = capture.read()
    if not ret:
        break  # Kilépés, ha nincs több képkocka

    # --- Inpainting csak a tisztított maszkon ---
    inpainted_frame = cv.inpaint(frame, dilated_mask, inpaintRadius=10, flags=cv.INPAINT_TELEA)

    # --- Képkockák megjelenítése ---
    cv.imshow("Original Video", frame)
    cv.imshow("Cleaned Mask", cleaned_mask)
    cv.imshow("Dilated Mask", dilated_mask)  # Megmutatja, hogy mit használ az inpainting
    cv.imshow("Inpainted Video", inpainted_frame)

    # --- Kilépés billentyűparanccsal ---
    if cv.waitKey(30) & 0xFF == ord('q'):  # Kilépés 'q' gombbal
        break

# --- Erőforrások felszabadítása ---
capture.release()
cv.destroyAllWindows()
